"""SmishFuse: gated fusion of hand-crafted features and a fine-tuned DistilBERT [CLS].

f  : standardized hand-crafted features (n_f)        e : DistilBERT [CLS] embedding (768)
f' = MLP(f) -> d                                      e' = Linear(e) -> d
full (gated):   g = sigmoid(W [f; e] + b) in R^d,  h = g * f' + (1 - g) * e'
concat:         h = [f'; e']                       (no gate)
features_mlp:   h = f'                             (no encoder)
text_only:      h = e'                             (no features)
Two heads on h: 3 classes (ham / smishing / spam) and binary (ham vs malicious);
loss = CE_3 + binary_loss_weight * CE_bin. Prediction = argmax of the 3-class head.
"""
from __future__ import annotations

import torch
import torch.nn as nn

VARIANTS = {"smishfuse": "gated", "smishfuse_concat": "concat", "features_mlp": "features",
            "distilbert_text_only": "text"}


class SmishFuse(nn.Module):
    def __init__(self, mode: str, n_features: int, d: int, mlp_hidden: int, dropout: float, n_classes: int,
                 checkpoint: str | None = None):
        super().__init__()
        self.mode = mode
        self.uses_text = mode in ("gated", "concat", "text")
        self.uses_features = mode in ("gated", "concat", "features")
        if self.uses_text:
            from transformers import AutoModel

            self.encoder = AutoModel.from_pretrained(checkpoint)
            e_dim = self.encoder.config.hidden_size
            self.text_proj = nn.Linear(e_dim, d)
        if self.uses_features:
            self.feat_mlp = nn.Sequential(nn.Linear(n_features, mlp_hidden), nn.ReLU(), nn.Dropout(dropout),
                                          nn.Linear(mlp_hidden, d))
        if mode == "gated":
            self.gate = nn.Linear(n_features + e_dim, d)
        h_dim = 2 * d if mode == "concat" else d
        self.dropout = nn.Dropout(dropout)
        self.head3 = nn.Linear(h_dim, n_classes)
        self.head_bin = nn.Linear(h_dim, 2)

    def encoder_parameters(self):
        return list(self.encoder.named_parameters()) if self.uses_text else []

    def forward(self, features=None, input_ids=None, attention_mask=None):
        g = None
        if self.uses_text:
            e = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
            e_p = self.text_proj(e)
        if self.uses_features:
            f_p = self.feat_mlp(features)
        if self.mode == "gated":
            g = torch.sigmoid(self.gate(torch.cat([features.float(), e.float()], dim=1)))
            h = g * f_p + (1 - g) * e_p
        elif self.mode == "concat":
            h = torch.cat([f_p, e_p], dim=1)
        elif self.mode == "features":
            h = f_p
        else:
            h = e_p
        h = self.dropout(h)
        return self.head3(h), self.head_bin(h), g
