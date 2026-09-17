"""SD-RPN stage1 FULL training for Gemma-4-12B-IT (NCI Gadi, 4x A100-80GB).

Scales qwen_src/gemma4_unified/train_stage1_smoke.py (see SMOKE_REPORT.md)
to the full 172k corpus via torchrun DDP:

  torchrun --nproc_per_node=4 train_stage1_full.py --image-root ...

Same architecture as the smoke: truncated base (layers 0..29), twig K=27
T=3 warm-started from layers 27/28/29, base frozen, RoI BCE vs the ONLINE
pseudo-label, eager attention. Additions:
  * DDP (twig-only grads; no_sync during accumulation; broadcast_buffers off)
  * fg-recall / fg-precision monitoring (loss is BG-dominated → uninformative)
  * RoI-vs-label viz PNGs every --viz-every steps (rank 0)
  * atomic delta checkpoints every --save-every steps + resume support
    (resume reloads twig + optimizer + step count; data order restarts from
    a fresh epoch shuffle — acceptable for a 1-epoch run killed mid-flight).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.environ.get(
    "VISIONRL2_REPO", os.getcwd()))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from transformers import AutoConfig, AutoProcessor

from qwen_src.gemma4_unified.modeling_gemma4_unified_batch import (
    Gemma4UnifiedForConditionalGeneration,
)
from qwen_src.gemma4_unified.data_gemma_stage1 import (
    GemmaStage1Dataset,
    make_collate_padded,
)

MODEL_ID = "google/gemma-4-12B-it"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="data/sdrpn/gemma4_12b/gemma12b_it_560_train.jsonl")
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--out-dir", default="output/sdrpn/gemma4-12b-sdrpn-K27T3-delta")
    ap.add_argument("--micro-batch", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--keep-layers", type=int, default=30)
    ap.add_argument("--twig-K", type=int, default=27)
    ap.add_argument("--twig-T", type=int, default=3)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scheduler", default="constant", choices=["constant", "cosine"])
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--viz-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=-1)
    # ---- stage2.5 twig re-calibration additions ----
    ap.add_argument("--base-ckpt", default=MODEL_ID,
                    help="base checkpoint dir (stage2.5: the LoRA-merged "
                         "post-SFT full ckpt)")
    ap.add_argument("--twig-init-from-delta", default=None,
                    help="load twig from this delta (SKIPS the layer-copy "
                         "re-init — the qwen is_2_5_stage=True analog)")
    ap.add_argument("--max-samples", type=int, default=-1,
                    help="subsample the pool (stage2.5: 25000)")
    ap.add_argument("--resume", default="auto",
                    help="'auto' (out_dir/resume_latest.pt if present), 'none', or a path")
    return ap.parse_args()


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


_total_steps_for_sched = 10**9

def main():
    args = parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(args.seed + rank)

    def p0(*a):
        if rank == 0:
            print(*a, flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    viz_dir = os.path.join(args.out_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)

    config = AutoConfig.from_pretrained(args.base_ckpt)
    tc = config.text_config
    tc.num_hidden_layers = args.keep_layers
    tc.enable_twig = True
    tc.twig_K = args.twig_K
    tc.twig_T = args.twig_T
    tc.opl_max_layer = 29
    tc.online_pseudo_label = True
    tc.roi_binary_coeff = float(os.environ.get("ROI_BINARY_COEFF", "0.25"))
    tc.bg_coff = float(os.environ.get("BG_COFF", "0.05"))

    p0(f"[full] loading {MODEL_ID} on {world} ranks "
       f"(layers 0..{args.keep_layers - 1}, twig K={args.twig_K} T={args.twig_T})")
    t0 = time.time()
    model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
        args.base_ckpt, config=config, dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    p0(f"[full] loaded in {time.time() - t0:.1f}s")
    if args.twig_init_from_delta:
        # stage2.5 (qwen is_2_5_stage analog): CONTINUE the trained twig —
        # no layer-copy re-init.
        ck = torch.load(args.twig_init_from_delta, map_location="cpu",
                        weights_only=False)
        model.model.language_model.twig_layers.load_state_dict({
            k.replace("model.language_model.twig_layers.", ""): v
            for k, v in ck["delta_state_dict"].items()
            if k.startswith("model.language_model.twig_layers.")})
        p0(f"[full] twig continued from {args.twig_init_from_delta} "
           f"(steps={ck.get('meta', {}).get('steps')}); re-init SKIPPED")
    else:
        model.load_twig_weights_from_original_model()

    for p in model.parameters():
        p.requires_grad_(False)
    twig = model.model.language_model.twig_layers
    for p in twig.parameters():
        p.requires_grad_(True)
    n_train = sum(p.numel() for p in twig.parameters())
    p0(f"[full] trainable {n_train / 1e6:.1f}M params")

    model.to(device)
    model.train()
    model._roi_stats_buffer = collections.deque(maxlen=8 * args.micro_batch * args.accum)

    # ---- resume (twig + optimizer + step), before DDP wrap ----
    start_step = 0
    resume_path = None
    if args.resume == "auto":
        cand = os.path.join(args.out_dir, "resume_latest.pt")
        resume_path = cand if os.path.exists(cand) else None
    elif args.resume not in ("none", ""):
        resume_path = args.resume
    resume_opt_state = None
    if resume_path is not None:
        ck = torch.load(resume_path, map_location="cpu", weights_only=False)
        twig.load_state_dict({
            k.replace("model.language_model.twig_layers.", ""): v
            for k, v in ck["delta_state_dict"].items()
        })
        resume_opt_state = ck.get("optimizer")
        start_step = int(ck["meta"].get("steps", 0))
        p0(f"[full] resumed twig from {resume_path} at step {start_step}")

    ddp = DDP(model, device_ids=[local_rank], broadcast_buffers=False,
              find_unused_parameters=False, gradient_as_bucket_view=True)

    processor = AutoProcessor.from_pretrained(args.base_ckpt)
    ds = GemmaStage1Dataset(
        args.jsonl, processor, image_root=args.image_root,
        max_samples=args.max_samples,
        shuffle_seed=(args.seed if args.max_samples > 0 else None),
        # subsample needs a seeded shuffle first so the 25k cut is random;
        # the DistributedSampler still reshuffles per epoch
    )
    sampler = DistributedSampler(ds, num_replicas=world, rank=rank,
                                 shuffle=True, seed=args.seed, drop_last=True)
    pad_id = processor.tokenizer.pad_token_id
    pad_id = 0 if pad_id is None else int(pad_id)
    dl = DataLoader(
        ds, batch_size=args.micro_batch, sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=make_collate_padded(pad_id),
        pin_memory=True, drop_last=True, persistent_workers=True,
        prefetch_factor=4,
    )

    opt = torch.optim.AdamW(twig.parameters(), lr=args.lr,
                            betas=(0.9, 0.999), weight_decay=0.0)
    if resume_opt_state is not None:
        try:
            opt.load_state_dict(resume_opt_state)
            p0("[full] optimizer state restored")
        except Exception as e:
            p0(f"[full] optimizer restore failed ({e}); fresh optimizer")

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / max(1, args.warmup)
        if getattr(args, "scheduler", "constant") == "cosine":
            import math as _m
            prog = (step - args.warmup) / max(1, _total_steps_for_sched - args.warmup)
            return args.lr * 0.5 * (1.0 + _m.cos(_m.pi * min(1.0, prog)))
        return args.lr

    micro_per_step = args.accum
    steps_per_epoch = len(dl) // micro_per_step
    total_steps = steps_per_epoch * args.epochs
    global _total_steps_for_sched
    _total_steps_for_sched = total_steps
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    eff_batch = args.micro_batch * args.accum * world
    p0(f"[full] {len(ds)} samples | eff_batch={eff_batch} | "
       f"{steps_per_epoch} steps/epoch | total {total_steps} steps | "
       f"start_step={start_step}")

    tensor_keys = ("input_ids", "attention_mask", "labels", "pixel_values",
                   "image_position_ids", "mm_token_type_ids")

    def save_ckpt(step, final=False):
        if rank != 0:
            return
        delta = {
            f"model.language_model.twig_layers.{k}": v.detach().cpu()
            for k, v in twig.state_dict().items()
        }
        meta = {
            "model_id": MODEL_ID, "twig_K": args.twig_K, "twig_T": args.twig_T,
            "keep_layers": args.keep_layers, "steps": step, "lr": args.lr,
            "eff_batch": eff_batch, "seed": args.seed,
            "note": "SD-RPN stage1 full; merge into full 48-layer ckpt as "
                    "twig_layers. Truncated base never saved.",
        }
        tag = "final" if final else f"step{step:06d}"
        atomic_save({"delta_state_dict": delta, "meta": meta},
                    os.path.join(args.out_dir, f"twig_delta_{tag}.pt"))
        atomic_save({"delta_state_dict": delta, "meta": meta,
                     "optimizer": opt.state_dict()},
                    os.path.join(args.out_dir, "resume_latest.pt"))
        print(f"[ckpt] saved twig_delta_{tag}.pt + resume_latest.pt", flush=True)

    def dump_viz(step, stats):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
            picks = [s for s in stats if not s["all_ignore"]][-4:]
            if not picks:
                return
            fig, axes = plt.subplots(2, len(picks),
                                     figsize=(3.0 * len(picks), 6.2),
                                     squeeze=False)
            for c, s in enumerate(picks):
                lg = s["label_grid"].copy()
                lg_show = np.where(lg == -100, 0.5, lg)  # ignore → gray
                axes[0][c].imshow(lg_show, cmap="coolwarm", vmin=0, vmax=1,
                                  interpolation="nearest")
                axes[0][c].set_title(f"label {s['mode']} fg={s['n_fg']}",
                                     fontsize=8)
                axes[1][c].imshow(s["score_grid"], cmap="viridis", vmin=0,
                                  vmax=1, interpolation="nearest")
                axes[1][c].set_title(
                    f"pred tp={s['tp']}/{s['n_pred']}", fontsize=8)
                for ax in (axes[0][c], axes[1][c]):
                    ax.axis("off")
            fig.suptitle(f"step {step}: online label (top) vs sigmoid RoI "
                         f"score (bottom)", fontsize=9)
            fig.tight_layout()
            fig.savefig(os.path.join(viz_dir, f"roi_step{step:06d}.png"),
                        dpi=120)
            plt.close(fig)
        except Exception as e:
            print(f"[viz] failed: {e}", flush=True)

    step = start_step
    t_start = time.time()
    micro_losses = []
    win = {"tp": 0, "n_fg": 0, "n_pred": 0, "n_samples": 0, "n_ignore": 0,
           "n_bg": 0}
    log_t0 = time.time()
    log_step0 = step
    opt.zero_grad(set_to_none=True)
    done = False

    for epoch in range(args.epochs + 1):  # +1 slack when resuming mid-epoch
        if done:
            break
        sampler.set_epoch(epoch)
        micro_in_step = 0
        for batch in dl:
            for k in tensor_keys:
                if k in batch:
                    batch[k] = batch[k].to(device, non_blocking=True)
            micro_in_step += 1
            sync_now = micro_in_step == args.accum
            ctx = ddp.no_sync() if not sync_now else _nullctx()
            with ctx:
                out = ddp(**batch, use_cache=False)
                loss = out.loss
                if loss is None:  # defensive: keep DDP sync counters aligned
                    loss = out.twig_hidden_states.sum() * 0.0
                (loss / args.accum).backward()
            micro_losses.append(float(loss.detach()))

            # drain per-sample stats into the window aggregate
            while model._roi_stats_buffer:
                s = model._roi_stats_buffer.popleft()
                win["n_samples"] += 1
                if s["all_ignore"]:
                    win["n_ignore"] += 1
                else:
                    win["tp"] += s["tp"]
                    win["n_fg"] += s["n_fg"]
                    win["n_pred"] += s["n_pred"]
                    win["n_bg"] += s["n_bg"]
                last_stats = s
                if rank == 0:
                    _viz_keep.append(s)

            if not sync_now:
                continue
            micro_in_step = 0
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            gnorm = torch.nn.utils.clip_grad_norm_(twig.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0 and rank == 0:
                mean_loss = sum(micro_losses) / max(1, len(micro_losses))
                micro_losses.clear()
                rec = win["tp"] / max(1, win["n_fg"])
                prec = win["tp"] / max(1, win["n_pred"])
                ign_frac = win["n_ignore"] / max(1, win["n_samples"])
                dt = time.time() - log_t0
                sps = (step - log_step0) / max(1e-9, dt)
                eta_h = (total_steps - step) / max(1e-9, sps) / 3600
                print(f"[step {step:5d}/{total_steps}] loss={mean_loss:.4f} "
                      f"fg_recall={rec:.3f} fg_prec={prec:.3f} "
                      f"fg/sample={win['n_fg'] / max(1, win['n_samples'] - win['n_ignore']):.1f} "
                      f"ignore_frac={ign_frac:.2f} gnorm={float(gnorm):.2f} "
                      f"lr={lr_at(step - 1):.2e} {sps:.2f} steps/s "
                      f"ETA={eta_h:.2f}h", flush=True)
                win = {k: 0 for k in win}
                log_t0 = time.time()
                log_step0 = step

            if step % args.viz_every == 0 and rank == 0:
                dump_viz(step, list(_viz_keep))

            if step % args.save_every == 0:
                dist.barrier()
                save_ckpt(step)

            if step >= total_steps:
                done = True
                break

    dist.barrier()
    save_ckpt(step, final=True)
    p0(f"[full] DONE at step {step} in {(time.time() - t_start) / 3600:.2f}h")
    dist.destroy_process_group()


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


_viz_keep = collections.deque(maxlen=8)


if __name__ == "__main__":
    main()
