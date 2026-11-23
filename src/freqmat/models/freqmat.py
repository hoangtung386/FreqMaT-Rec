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
        if cfg.extrap:
            self.extrap_dt = nn.Parameter(torch.zeros(()))  # 0 -> identity at init
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

    def _readout(self, x: torch.Tensor, item_seq_len: torch.Tensor, extrap: bool = False) -> torch.Tensor:
        """State at the last real item ``(B, D)``; optionally extrapolated one step along its causal velocity."""
        q = self.gather_indexes(x, item_seq_len - 1)
        if extrap:
            prev = self.gather_indexes(x, (item_seq_len - 2).clamp_min(0))
            q = q + self.extrap_dt * (q - prev) * (item_seq_len > 1).unsqueeze(-1).to(q.dtype)
        return q

    def forward(self, item_seq, item_seq_len, timestamps=None):
        x = self.hidden_states(item_seq, timestamps)[-1]
        return self._readout(x, item_seq_len, self.cfg.extrap)

    def _encode(self, interaction) -> torch.Tensor:
        """Sequence representation ``(B, D)`` for a RecBole interaction batch."""
        return self.forward(interaction[self.ITEM_SEQ], interaction[self.ITEM_SEQ_LEN], self._timestamps(interaction))

    def _aux_loss(self):
        if not self.moe_aux_weight:
            return 0.0
        return self.moe_aux_weight * sum(layer.ffn.aux_loss for layer in self.layers if layer.use_moe)

    def calculate_loss(self, interaction):
        seq, seq_len = interaction[self.ITEM_SEQ], interaction[self.ITEM_SEQ_LEN]
        states = self.hidden_states(seq, self._timestamps(interaction))
        out = self._readout(states[-1], seq_len, self.cfg.extrap)
        pos = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            pos_score = (out * self.item_embedding(pos)).sum(-1)
            neg_score = (out * self.item_embedding(interaction[self.NEG_ITEM_ID])).sum(-1)
            rec_loss = self.loss_fct(pos_score, neg_score)
        else:
            rec_loss = self.loss_fct(out @ self.item_embedding.weight.T, pos)
        if self.cfg.deep_sup_active and len(states) > 2:
            inner = states[1:-1]
            weights = self.item_embedding.weight.T
            deep = sum(self.loss_fct(self._readout(h, seq_len) @ weights, pos) for h in inner) / len(inner)
            rec_loss = rec_loss + self.cfg.deep_sup_weight * deep
        return rec_loss + self._aux_loss()

    def predict(self, interaction):
        return (self._encode(interaction) * self.item_embedding(interaction[self.ITEM_ID])).sum(-1)

    def full_sort_predict(self, interaction):
        return self._encode(interaction) @ self.item_embedding.weight.T
