"""FreqMaT: Frequency-Aware Mamba-Transformer for Sequential Recommendation (RecBole model)."""

import torch
import torch.nn as nn
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss

from freqmat.models.config import FreqMaTConfig
from freqmat.models.layer import FreqMaTLayer
from freqmat.models.modules import time_graph_basis


class FreqMaT(SequentialRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.cfg = FreqMaTConfig.from_recbole(config)
        cfg = self.cfg
        self.loss_type = cfg.loss_type
        self.time_key = config["TIME_FIELD"] + config["LIST_SUFFIX"]
        self.needs_time = cfg.use_freq and cfg.freq_type == "gft"
        self.moe_aux_weight = cfg.moe_aux_weight if cfg.ffn_type == "moe" else 0.0
        self.item_embedding = nn.Embedding(self.n_items, cfg.hidden_size, padding_idx=0)
        self.emb_drop = nn.Dropout(cfg.hidden_dropout_prob)
        self.emb_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.layers = nn.ModuleList([FreqMaTLayer(cfg) for _ in range(cfg.n_layers)])
        if cfg.loss_type == "BPR":
            if not config["train_neg_sample_args"]:
                raise ValueError("loss_type=BPR needs train_neg_sample_args (negative sampling) in the config")
            self.loss_fct = BPRLoss()
        else:
            self.loss_fct = nn.CrossEntropyLoss()
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(0.0, 0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _timestamps(self, interaction):
        return interaction[self.time_key] if self.time_key in interaction else None  # noqa: SIM401 (Interaction has no .get)

    def hidden_states(self, item_seq: torch.Tensor, timestamps: torch.Tensor | None = None) -> list:
        """``[embedding output, layer 1 output, ..., layer L output]``, each ``(B, N, D)``."""
        mask = item_seq != 0
        x = self.emb_norm(self.emb_drop(self.item_embedding(item_seq)))
        x = x * mask.unsqueeze(-1)  # zero padding (padding sits at the end of each sequence in RecBole)
        basis = None
        if self.needs_time:
            if timestamps is None:
                raise ValueError(
                    "freq_type=gft needs the timestamp list field: keep `timestamp` in load_col.inter of the config"
                )
            basis = time_graph_basis(timestamps, mask, self.cfg.freq_cutoff, self.cfg.gft_gamma)
        states = [x]
        for layer in self.layers:
            x = layer(x, mask, basis)
            states.append(x)
        return states

    def forward(self, item_seq, item_seq_len, timestamps=None):
        x = self.hidden_states(item_seq, timestamps)[-1]
        return self.gather_indexes(x, item_seq_len - 1)

    def _encode(self, interaction) -> torch.Tensor:
        """Sequence representation ``(B, D)`` for a RecBole interaction batch."""
        return self.forward(interaction[self.ITEM_SEQ], interaction[self.ITEM_SEQ_LEN], self._timestamps(interaction))

    def _aux_loss(self):
        if not self.moe_aux_weight:
            return 0.0
        return self.moe_aux_weight * sum(layer.ffn.aux_loss for layer in self.layers if layer.use_moe)

    def calculate_loss(self, interaction):
        out = self._encode(interaction)
        pos = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            pos_score = (out * self.item_embedding(pos)).sum(-1)
            neg_score = (out * self.item_embedding(interaction[self.NEG_ITEM_ID])).sum(-1)
            rec_loss = self.loss_fct(pos_score, neg_score)
        else:
            rec_loss = self.loss_fct(out @ self.item_embedding.weight.T, pos)
        return rec_loss + self._aux_loss()

    def predict(self, interaction):
        return (self._encode(interaction) * self.item_embedding(interaction[self.ITEM_ID])).sum(-1)

    def full_sort_predict(self, interaction):
        return self._encode(interaction) @ self.item_embedding.weight.T
