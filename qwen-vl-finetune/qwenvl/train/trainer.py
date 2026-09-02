"""Trainer patch used by ``train_qwen.py``.

Importing this module installs a ``create_optimizer`` on ``transformers.Trainer``
that splits the trainable parameters into a weight-decay group and a no-decay
group (biases and norm weights). Only parameters with ``requires_grad=True``
(the SD-RPN twig blocks) end up in the optimizer.
"""

from transformers import Trainer


def create_optimizer(self):

    opt_model = self.model

    if self.optimizer is None:
        decay_parameters = self.get_decay_parameter_names(opt_model)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
            self.args
        )
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    return self.optimizer


# --- APPLY PATCHES ---
Trainer.create_optimizer = create_optimizer
