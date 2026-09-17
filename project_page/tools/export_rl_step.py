#!/usr/bin/env python
"""Export one region-level RL training step per sample for the project page.

For each selected RL-pool sample this reproduces, with the released trainer's
own building blocks, the quantities of Sec. 3 of the paper on the SD-RPN
(Phase-A) policy and dumps them as JSON + thumbnails:

  * policy map P_theta, reference map, regions R_1..R_K (top-K components)
  * subtractive group: intact mask + singleton removals, masked images,
    reader P_phi / h_phi per action, contributions Delta_k, control (placebo)
    probes and the margin b, advantages A_k, removal policy pi_sub
  * additive group: supplementary regions from the cached evidence maps,
    Delta_j, advantages, inclusion log-likelihood
  * the final RL policy's map on the same sample (twig weights swapped in)

Modes:
  --scan N           : step through the first N pool rows (Phase-A only, no
                       thumbnails) and write <out>/candidates.jsonl with
                       per-sample summary stats for case selection.
  --indices i,j,k    : full dump for those pool row indices into <out>/<idx>/.

Run from the VisionRL2 root with PYTHONPATH=<root>:<root>/qwen-vl-finetune.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# --------------------------------------------------------------------- setup
REPO = Path(__file__).resolve().parents[2]
for p in (REPO, REPO / "qwen-vl-finetune", REPO / "lmms-eval"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def log(*a):
    print("[export]", *a, file=sys.stderr, flush=True)


def build_model(ckpt: str, twig_K: int, twig_T: int, device: str):
    import transformers
    from qwenvl.train.region_level_grpo.train_phase_b1 import _load_policy_with_twig
    model_args = types.SimpleNamespace(model_name_or_path=ckpt, twig_K=twig_K,
                                       twig_T=twig_T, roi_loss="bce")
    training_args = types.SimpleNamespace(bf16=True, cache_dir=None)
    model = _load_policy_with_twig(model_args, training_args,
                                   attn_implementation="sdpa")
    model.config.use_cache = False
    for cfg in (model.config, getattr(model.config, "text_config", None)):
        if cfg is not None:
            cfg.roi_score_with_rope = True
            cfg.roi_score_query_mode = "last_prompt"
    model.eval().to(device)
    processor = transformers.AutoProcessor.from_pretrained(ckpt)
    processor.image_processor.min_pixels = 262144
    processor.image_processor.max_pixels = 589824
    return model, processor


def load_rl_twig(model, rl_ckpt: str):
    """Swap the RL checkpoint's twig weights into the loaded policy."""
    from safetensors.torch import load_file
    d = Path(rl_ckpt)
    files = sorted(d.glob("*.safetensors"))
    sd = {}
    for f in files:
        part = load_file(str(f))
        sd.update({k: v for k, v in part.items() if "twig" in k})
    missing, unexpected = model.load_state_dict(sd, strict=False)
    n_loaded = len(sd) - len([u for u in unexpected if u in sd])
    log(f"RL twig: {len(sd)} twig tensors in ckpt, unexpected={len(unexpected)}")
    return len(sd)


def policy_forward(model, ref_twig, collator, item, device):
    """One policy forward mirroring RegionLevelGRPOTrainer.compute_loss."""
    from qwenvl.train.region_level_grpo import trainer as T
    inputs = collator([item])
    for k in (T.KEY_PIL_IMAGES, T.KEY_QUESTIONS, T.KEY_GOLD_ANSWERS,
              T.KEY_P_REFS, T.KEY_P_REF_BINARIES, T.KEY_EV_MAPS):
        inputs.pop(k, None)
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    cfg = model.config
    cfg.return_per_head_score = True
    lm_node = getattr(model, "model", model)
    lm_node = getattr(lm_node, "language_model", lm_node)
    saved = getattr(lm_node, "roi_enable2stage", None)
    if saved is not None:
        lm_node.roi_enable2stage = False
    with torch.no_grad():
        out = model(**inputs)
    if saved is not None:
        lm_node.roi_enable2stage = saved
    phs = (getattr(out, "per_head_scores", None) or [None])[0]
    feat_hw = (getattr(out, "feat_hw", None) or [None])[0]
    ref_phs = None
    if ref_twig is not None and getattr(out, "pre_twig_ctx", None) is not None:
        ctx = out.pre_twig_ctx
        ref_twig.to(device=ctx["hidden_states"].device, dtype=ctx["hidden_states"].dtype)
        with torch.no_grad():
            ref_phs = ref_twig.compute_per_head_scores(
                pre_twig_ctx=ctx, input_ids=inputs.get("input_ids"),
                labels=inputs.get("labels"), image_token_id=model.config.image_token_id,
                image_grid_thw=inputs.get("image_grid_thw"), apply_rope=True,
                query_mode="last_prompt")[0]
    return phs, feat_hw, ref_phs


def logit_clip_heights(log_probs: torch.Tensor, g0_idx: int, delta: float = 5.0):
    eps = 1e-6
    p = torch.exp(log_probs).clamp(max=1.0 - eps)
    lg = log_probs - torch.log1p(-p)
    g0 = lg[g0_idx]
    return g0 + torch.clamp(lg - g0, min=-delta, max=delta), lg


def mask_to_list(m: np.ndarray):
    return [[int(v) for v in row] for row in np.asarray(m).astype(np.uint8)]


def thumb(pil: Image.Image, path: Path, max_side: int = 768, q: int = 88):
    im = pil.convert("RGB")
    s = max(im.size)
    if s > max_side:
        im = im.resize((round(im.width * max_side / s), round(im.height * max_side / s)),
                       Image.BICUBIC)
    im.save(path, quality=q)
    return im.size


def run_step(model, ref_twig, reward_model, collator, item, device, cfg, want_images):
    """Compute the subtractive + additive groups for one sample (Phase-A policy)."""
    from qwenvl.train.region_level_grpo.components import (
        extract_components, rank_top_r, score_component_torch)
    from qwenvl.train.region_level_grpo.actions import enumerate_removal_actions
    from qwenvl.train.region_level_grpo.policy import compute_log_pi
    from qwenvl.train.region_level_grpo.losses import policy_gradient_loss_empty_baseline
    from qwenvl.train.region_level_grpo.trainer import _build_masked_pils, _build_placebo_blobs
    from scipy.ndimage import label as cc_label

    phs, feat_hw, ref_phs = policy_forward(model, ref_twig, collator, item, device)
    if phs is None:
        return {"error": "no per-head scores"}
    Hg, Wg = int(feat_hw[0]), int(feat_hw[1])
    n_grid = Hg * Wg
    Z = phs.float().mean(0).view(Hg, Wg)
    P = torch.sigmoid(Z)
    res = {"grid": [Hg, Wg], "P": P.cpu().tolist(),
           "P_max": float(P.max()), "P_mean": float(P.mean())}
    if ref_phs is not None:
        Pref = torch.sigmoid(ref_phs.float().mean(0).view(Hg, Wg))
        res["P_ref"] = Pref.cpu().tolist()

    comps = extract_components(
        P.detach(), Z.detach(), smooth_kernel=cfg["smooth_kernel"],
        smooth_sigma=cfg["smooth_sigma"], threshold_mode="peak_ratio",
        fixed_threshold=cfg["fixed_threshold"], ratio_thresh=cfg["ratio_thresh"],
        peak_fraction=cfg["peak_fraction"], min_gate=cfg["min_gate"],
        score_p=1.0, score_beta=1.0, score_gamma=0.6, connectivity=1)
    top = rank_top_r(comps, R=cfg["R_max"])
    K = len(top)
    res["K"] = K
    res["regions"] = [{"id": i, "area": int(c.area), "bbox": [int(x) for x in c.bbox],
                       "z": float(score_component_torch(Z, torch.from_numpy(c.mask).to(device),
                                                        p=1.0, beta=1.0, gamma=0.6)),
                       "mask": mask_to_list(c.mask)} for i, c in enumerate(top)]
    src = item["pil_image"].convert("RGB")
    q, gold = item["question"], item["gold_answer"]
    imgs = {}

    # ------------------------------------------------ subtractive group
    if K >= 2:
        comp_scores = torch.stack([score_component_torch(
            Z, torch.from_numpy(c.mask).to(device), p=1.0, beta=1.0, gamma=0.6) for c in top])
        actions = enumerate_removal_actions(top, singleton_only=True)
        log_pi = compute_log_pi(comp_scores, actions, n_grid, T=1.0, beta_keep=1.0,
                                gamma_keep=0.6)
        keeps = np.stack([a.keep_mask for a in actions], 0)
        pils = _build_masked_pils(src, keeps)
        with torch.no_grad():
            lp = reward_model.compute_logprobs(masked_images=pils, question=q,
                                               gold_answer=gold, device=device,
                                               reduction="mean").float()
        empty_idx = next(i for i, a in enumerate(actions) if not a.discard)
        heights, lg = logit_clip_heights(lp, empty_idx, cfg["clip_delta"])
        # placebo / control probes
        union = actions[empty_idx].keep_mask.astype(np.uint8)
        drop_areas = [int(union.sum() - a.keep_mask.sum()) for a in actions if len(a.discard) == 1]
        seed = zlib.crc32((str(q) + "|" + str(gold)).encode()) & 0x7FFFFFFF
        blobs = _build_placebo_blobs(P.detach().float().cpu().numpy(), union,
                                     max(1, int(np.median(drop_areas))), 2,
                                     cfg["placebo_p_thresh"], seed) if drop_areas else []
        bar = None
        probes = []
        if blobs:
            pk = np.stack([np.maximum(union, b) for b in blobs], 0)
            ppils = _build_masked_pils(src, pk)
            with torch.no_grad():
                plp = reward_model.compute_logprobs(masked_images=ppils, question=q,
                                                    gold_answer=gold, device=device,
                                                    reduction="mean").float()
            pp = torch.exp(plp).clamp(max=1 - 1e-6)
            plg = plp - torch.log1p(-pp)
            g0 = lg[empty_idx]
            ph = g0 + torch.clamp(plg - g0, min=-cfg["clip_delta"], max=cfg["clip_delta"])
            bar = min(cfg["kappa"] * float((heights[empty_idx] - ph).abs().max()), cfg["bar_max"])
            for j, b in enumerate(blobs):
                probes.append({"id": j, "mask": mask_to_list(b), "area": int(b.sum()),
                               "p": float(torch.exp(plp[j])), "h": float(ph[j]),
                               "abs_dh": float((heights[empty_idx] - ph[j]).abs())})
                if want_images:
                    imgs[f"probe_{j}.jpg"] = ppils[j]
        rewards = heights  # beta = 0, alpha = 0 in the released recipe
        adv = None
        if bar is not None:
            keep_mask = torch.ones(len(actions), dtype=torch.bool)
            keep_mask[empty_idx] = False
            r_empty = rewards[empty_idx] - bar
            pol = policy_gradient_loss_empty_baseline(
                log_pi[keep_mask].detach(), rewards[keep_mask], r_empty, std_regularizer=1.0)
            adv_full = torch.zeros(len(actions))
            adv_full[keep_mask] = pol["advantage"].detach().float().cpu()
            adv = adv_full
        acts = []
        for i, a in enumerate(actions):
            d = {"id": i, "discard": list(a.discard), "keep_frac": float(a.keep_mask.mean()),
                 "p": float(torch.exp(lp[i])), "logp": float(lp[i]), "h": float(heights[i]),
                 "log_pi": float(log_pi[i]), "pi": float(torch.exp(log_pi[i]))}
            if a.discard:
                d["delta"] = float(heights[empty_idx] - heights[i])
                if adv is not None:
                    d["adv"] = float(adv[i])
                    d["decision"] = "prune" if adv[i] > 0 else "keep"
            acts.append(d)
            if want_images:
                imgs[f"act_{i}.jpg"] = pils[i]
        if cfg.get("pairs") and 2 <= K <= 5:
            pairs, pmasks = [], []
            for i1 in range(K):
                for i2 in range(i1 + 1, K):
                    km = np.zeros((Hg, Wg), dtype=bool)
                    for kk in range(K):
                        if kk not in (i1, i2):
                            km |= top[kk].mask.astype(bool)
                    if km.sum() == 0:
                        continue
                    pairs.append((i1, i2)); pmasks.append(km.astype(np.uint8))
            if pairs:
                ppils2 = _build_masked_pils(src, np.stack(pmasks, 0))
                with torch.no_grad():
                    plp2 = reward_model.compute_logprobs(masked_images=ppils2, question=q,
                                                         gold_answer=gold, device=device,
                                                         reduction="mean").float()
                pp2 = torch.exp(plp2).clamp(max=1 - 1e-6)
                plg2 = plp2 - torch.log1p(-pp2)
                g0 = lg[empty_idx]
                ph2 = g0 + torch.clamp(plg2 - g0, min=-cfg["clip_delta"], max=cfg["clip_delta"])
                single = {a["discard"][0]: a["delta"] for a in acts if a["discard"]}
                res["pairs"] = [{"pair": list(pr), "delta_pair": float(heights[empty_idx] - ph2[t]),
                                 "delta_i": single.get(pr[0]), "delta_j": single.get(pr[1])}
                                for t, pr in enumerate(pairs)]
        res["sub"] = {"actions": acts, "empty_idx": empty_idx, "bar": bar,
                      "kappa": cfg["kappa"], "probes": probes,
                      "h_empty": float(heights[empty_idx]), "p_empty": float(torch.exp(lp[empty_idx]))}
    elif K == 1:
        res["sub"] = {"branch": "K=1 (BCE anchor only)"}
    else:
        res["sub"] = {"branch": "K=0 (KL anchor only)"}

    # ------------------------------------------------ additive group
    ev = item.get("ev_maps")
    if ev is not None and K >= 1:
        em = ev
        if em.dim() == 2:
            em = em.unsqueeze(0)
        em = em.to(torch.float32)
        if em.shape[-2:] != (Hg, Wg):
            em = F.interpolate(em.unsqueeze(1), size=(Hg, Wg), mode="nearest").squeeze(1)
        em = (em > 0.5).cpu().numpy().astype(np.uint8)
        cores = np.zeros((Hg, Wg), dtype=np.uint8)
        for c in top:
            cores = np.maximum(cores, c.mask.astype(np.uint8))
        cand = []
        for li in range(em.shape[0]):
            sl = ((em[li] > 0) & (cores == 0)).astype(np.uint8)
            if sl.sum() == 0:
                continue
            l2, n2 = cc_label(sl > 0)
            for ri in range(1, n2 + 1):
                rm = (l2 == ri).astype(np.uint8)
                if rm.sum() >= 1:
                    cand.append((rm, li))
        cand.sort(key=lambda t: -int(t[0].sum()))
        used = [False] * len(cand)
        supp, votes = [], []
        for i in range(len(cand)):
            if used[i]:
                continue
            seed_m, l0 = cand[i]
            used[i] = True
            members, layers = [seed_m], {l0}
            for j in range(i + 1, len(cand)):
                if used[j]:
                    continue
                mj, lj = cand[j]
                inter = int(((seed_m > 0) & (mj > 0)).sum())
                uni = int(((seed_m > 0) | (mj > 0)).sum())
                if uni > 0 and inter / uni > 0.5:
                    used[j] = True
                    members.append(mj)
                    layers.add(lj)
            st = np.stack(members, 0).astype(np.float32)
            merged = (st.mean(0) >= 0.5).astype(np.uint8)
            if merged.sum() == 0:
                merged = (st.sum(0) > 0).astype(np.uint8)
            supp.append(merged)
            votes.append(len(layers))
        order = sorted(range(len(supp)), key=lambda i: -int(supp[i].sum()))[:cfg["supp_k_max"]]
        supp = [supp[i] for i in order]
        votes = [votes[i] for i in order]
        res["ev_layers"] = int(em.shape[0])
        res["ev_union"] = mask_to_list((em.sum(0) > 0).astype(np.uint8))
        if supp:
            su = np.zeros((Hg, Wg), dtype=np.uint8)
            for m in supp:
                su = np.maximum(su, m)
            full = np.maximum(cores, su)
            keep_list = [full] + [((full > 0) & (m == 0)).astype(np.uint8) for m in supp]
            kn = np.stack(keep_list, 0).astype(np.uint8)
            spils = _build_masked_pils(src, kn)
            with torch.no_grad():
                slp = reward_model.compute_logprobs(masked_images=spils, question=q,
                                                    gold_answer=gold, device=device,
                                                    reduction="mean").float()
            sh, _ = logit_clip_heights(slp, 0, cfg["clip_delta"])
            c = (sh[0] - sh[1:]).detach()
            std = c.std() if len(supp) > 1 else torch.zeros(())
            sadv = c / (std + 1.0)
            logsig = F.logsigmoid(Z)
            adds = []
            for j, m in enumerate(supp):
                ell = float(logsig[torch.from_numpy(m.astype(bool)).to(device)].mean())
                adds.append({"id": j, "mask": mask_to_list(m), "area": int(m.sum()),
                             "votes": int(votes[j]), "p_drop": float(torch.exp(slp[j + 1])),
                             "h_drop": float(sh[j + 1]), "delta": float(c[j]),
                             "adv": float(sadv[j]), "ell_plus": ell,
                             "decision": "raise" if c[j] > 0 else "suppress"})
                if want_images:
                    imgs[f"supp_drop_{j}.jpg"] = spils[j + 1]
            if want_images:
                imgs["supp_full.jpg"] = spils[0]
            res["add"] = {"supp": adds, "p_full": float(torch.exp(slp[0])), "h_full": float(sh[0]),
                          "full_mask": mask_to_list(full)}
        else:
            res["add"] = {"branch": "no supplementary candidates"}
    return res, imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--ckpt-pa", required=True)
    ap.add_argument("--ckpt-rl", default=None)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--ev-maps-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scan", type=int, default=0)
    ap.add_argument("--scan-start", type=int, default=0)
    ap.add_argument("--indices", default="")
    ap.add_argument("--kappa", type=float, default=1.25)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--pairs", action="store_true", help="also score pair removals (scan stats)")
    args = ap.parse_args()

    os.environ["DATASET_ROOT"] = args.dataset_root
    os.environ["EV_MAPS_ROOT"] = args.ev_maps_root
    from qwenvl.train.region_level_grpo.dataset import (
        RegionLevelGRPODataset, RegionLevelGRPOCollator, DS_IMAGE_ROOTS)
    from qwenvl.train.region_level_grpo.reward_model import RewardModel
    from qwenvl.train.region_level_grpo.ref_twig import load_ref_twig_from_policy

    cfg = dict(smooth_kernel=3, smooth_sigma=1.0, fixed_threshold=0.02, ratio_thresh=3.0,
               peak_fraction=0.3, min_gate=0.03, R_max=6, clip_delta=5.0,
               placebo_p_thresh=0.02, kappa=args.kappa, bar_max=1.0, supp_k_max=4,
               pairs=bool(args.pairs))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model, processor = build_model(args.ckpt_pa, 21, 3, args.device)
    ref_twig = load_ref_twig_from_policy(model)
    reward_model = RewardModel(model=model, processor=processor,
                               disable_thinking_prefix=True, skip_trailing_eos=True)
    ds = RegionLevelGRPODataset(filtered_jsonl=args.pool, image_roots=DS_IMAGE_ROOTS,
                                max_samples=0, online_p_ref=True)
    collator = RegionLevelGRPOCollator(processor=processor, fixed_threshold=0.02)
    log(f"model + dataset ready in {time.time() - t0:.0f}s, pool={len(ds)}")

    if args.scan:
        fo = open(out / "candidates.jsonl", "a")
        for idx in range(args.scan_start, min(len(ds), args.scan_start + args.scan)):
            try:
                item = ds[idx]
                res, _ = run_step(model, ref_twig, reward_model, collator, item, args.device, cfg, False)
                sub = res.get("sub", {})
                acts = sub.get("actions", [])
                summ = {"idx": idx, "sample_id": item["sample_id"], "dataset": item["dataset"],
                        "K": res["K"], "grid": res["grid"], "bar": sub.get("bar"),
                        "p_empty": sub.get("p_empty"),
                        "deltas": [a.get("delta") for a in acts if a.get("discard")],
                        "decisions": [a.get("decision") for a in acts if a.get("discard")],
                        "advs": [a.get("adv") for a in acts if a.get("discard")],
                        "pis": [a.get("pi") for a in acts if a.get("discard")],
                        "pi_empty": (acts[sub["empty_idx"]]["pi"] if acts else None),
                        "regions_area": [r["area"] for r in res.get("regions", [])],
                        "pairs": res.get("pairs"),
                        "supp": [(s["delta"], s["decision"]) for s in res.get("add", {}).get("supp", [])],
                        "question": item["question"][:120], "gold": item["gold_answer"][:60]}
                fo.write(json.dumps(summ) + "\n"); fo.flush()
                log(f"scan {idx}: K={res['K']} bar={sub.get('bar')} dec={summ['decisions']} supp={summ['supp']}")
            except Exception as e:  # noqa: BLE001
                log(f"scan {idx} FAILED: {e!r}")
        fo.close()

    if args.indices:
        idxs = [int(x) for x in args.indices.split(",") if x.strip()]
        results = {}
        for idx in idxs:
            item = ds[idx]
            d = out / f"{idx}"
            d.mkdir(exist_ok=True)
            res, imgs = run_step(model, ref_twig, reward_model, collator, item, args.device, cfg, True)
            res.update({"idx": idx, "sample_id": item["sample_id"], "dataset": item["dataset"],
                        "question": item["question"], "gold": item["gold_answer"],
                        "image_size": list(item["pil_image"].size)})
            res["src_size"] = thumb(item["pil_image"], d / "src.jpg", 1024)
            for name, im in imgs.items():
                thumb(im, d / name, 512)
            results[idx] = res
            log(f"dump {idx}: K={res['K']} bar={res.get('sub', {}).get('bar')}")
        if args.ckpt_rl:
            load_rl_twig(model, args.ckpt_rl)
            for idx in idxs:
                item = ds[idx]
                phs, feat_hw, _ = policy_forward(model, None, collator, item, args.device)
                Hg, Wg = int(feat_hw[0]), int(feat_hw[1])
                P = torch.sigmoid(phs.float().mean(0).view(Hg, Wg))
                from qwenvl.train.region_level_grpo.components import extract_components, rank_top_r
                comps = rank_top_r(extract_components(
                    P, torch.logit(P.clamp(1e-6, 1 - 1e-6)), smooth_kernel=3, smooth_sigma=1.0,
                    threshold_mode="peak_ratio", fixed_threshold=0.02, ratio_thresh=3.0,
                    peak_fraction=0.3, min_gate=0.03, score_p=1.0, score_beta=1.0,
                    score_gamma=0.6, connectivity=1), R=6)
                results[idx]["rl"] = {"P": P.cpu().tolist(), "P_max": float(P.max()),
                                      "regions": [{"id": i, "area": int(c.area),
                                                   "mask": mask_to_list(c.mask)} for i, c in enumerate(comps)]}
                # reader check on the RL map's foreground vs the Phase-A foreground
                from qwenvl.train.region_level_grpo.trainer import _build_masked_pils
                fg_rl = np.zeros((Hg, Wg), dtype=np.uint8)
                for c in comps:
                    fg_rl = np.maximum(fg_rl, c.mask.astype(np.uint8))
                fg_pa = np.zeros((Hg, Wg), dtype=np.uint8)
                for r in results[idx]["regions"]:
                    fg_pa = np.maximum(fg_pa, np.array(r["mask"], dtype=np.uint8))
                pils = _build_masked_pils(item["pil_image"].convert("RGB"), np.stack([fg_pa, fg_rl], 0))
                with torch.no_grad():
                    lp = reward_model.compute_logprobs(masked_images=pils, question=item["question"],
                                                       gold_answer=item["gold_answer"],
                                                       device=args.device, reduction="mean").float()
                results[idx]["rl"]["p_fg_pa"] = float(torch.exp(lp[0]))
                results[idx]["rl"]["p_fg_rl"] = float(torch.exp(lp[1]))
                results[idx]["rl"]["keep_frac"] = float(fg_rl.mean())
                results[idx]["keep_frac_pa"] = float(fg_pa.mean())
                thumb(pils[0], out / f"{idx}" / "fg_pa.jpg", 512)
                thumb(pils[1], out / f"{idx}" / "fg_rl.jpg", 512)
                log(f"rl {idx}: K_rl={len(comps)} p_fg_pa={results[idx]['rl']['p_fg_pa']:.3f} p_fg_rl={results[idx]['rl']['p_fg_rl']:.3f}")
        for idx, res in results.items():
            (out / f"{idx}" / "data.json").write_text(json.dumps(res), encoding="utf-8")
        log("done")


if __name__ == "__main__":
    main()
