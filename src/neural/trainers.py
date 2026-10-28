"""TextCNN / BiLSTM (word embeddings from scratch) and transformer fine-tuning.

One training loop for both kinds: AdamW/Adam, fp16 autocast + GradScaler,
evaluation on val after every epoch, early stopping on val macro F1 with
`patience`, the best epoch's weights are restored before predicting.
"""
from __future__ import annotations

import copy
import re
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
PAD, UNK = 0, 1


# ---------------------------------------------------------------- word models

def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def build_vocab(texts, min_freq: int) -> dict[str, int]:
    counts = Counter(t for text in texts for t in tokenize(text))
    words = sorted((w for w, c in counts.items() if c >= min_freq), key=lambda w: (-counts[w], w))
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for w in words:
        vocab[w] = len(vocab)
    return vocab


def encode_words(texts, vocab: dict, max_len: int) -> list[list[int]]:
    out = []
    for text in texts:
        ids = [vocab.get(t, UNK) for t in tokenize(text)][:max_len]
        out.append(ids or [UNK])
    return out


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, kernel_sizes, dropout, n_classes):
        super().__init__()
        self.kernel_sizes = [int(k) for k in str(kernel_sizes).split(",")]
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.convs = nn.ModuleList(nn.Conv1d(embed_dim, num_filters, k) for k in self.kernel_sizes)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(self.kernel_sizes), n_classes)

    def forward(self, input_ids, lengths=None):
        x = self.embedding(input_ids).transpose(1, 2)
        pooled = [torch.relu(conv(x)).amax(dim=2) for conv in self.convs]
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, dropout, n_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden_size, n_classes)

    def forward(self, input_ids, lengths):
        packed = pack_padded_sequence(self.embedding(input_ids), lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.fc(self.dropout(torch.cat([h[-2], h[-1]], dim=1)))


def word_collate(min_len: int):
    def collate(batch):
        seqs = [b[0] for b in batch]
        length = max(min_len, max(len(s) for s in seqs))
        ids = torch.full((len(seqs), length), PAD, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        out = {"input_ids": ids, "lengths": torch.tensor([len(s) for s in seqs], dtype=torch.long)}
        return out, torch.tensor([b[1] for b in batch], dtype=torch.long)
    return collate


class WordSetup:
    """Vocabulary + model factory for textcnn / bilstm."""

    def __init__(self, model_name: str, params: dict, train_texts, n_classes: int):
        self.model_name, self.params, self.n_classes = model_name, params, n_classes
        self.vocab = build_vocab(train_texts, params["min_freq"])
        ks = [int(k) for k in str(params.get("kernel_sizes", "1")).split(",")]
        self.collate = word_collate(max(ks))

    def encode(self, texts):
        return encode_words(texts, self.vocab, self.params["max_len"])

    def lengths(self, encoded):
        return [len(e) for e in encoded]

    def new_model(self) -> nn.Module:
        p = self.params
        if self.model_name == "textcnn":
            return TextCNN(len(self.vocab), p["embed_dim"], p["num_filters"], p["kernel_sizes"], p["dropout"],
                           self.n_classes)
        return BiLSTM(len(self.vocab), p["embed_dim"], p["hidden_size"], p["num_layers"], p["dropout"],
                      self.n_classes)

    def optimizer(self, model, n_steps: int):
        return torch.optim.Adam(model.parameters(), lr=self.params["lr"]), None

    def save(self, model, path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"model_name": self.model_name, "params": self.params, "n_classes": self.n_classes,
                    "vocab": self.vocab, "state_dict": model.state_dict()}, path / "model.pt")


# ---------------------------------------------------------------- transformers

class HFClassifier(nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.hf = hf_model

    def forward(self, input_ids, attention_mask):
        return self.hf(input_ids=input_ids, attention_mask=attention_mask).logits


class TransformerSetup:
    _tokenizers: dict = {}

    def __init__(self, checkpoint: str, params: dict, fixed: dict, n_classes: int):
        from transformers import AutoTokenizer

        self.checkpoint, self.params, self.fixed, self.n_classes = checkpoint, params, fixed, n_classes
        if checkpoint not in self._tokenizers:
            self._tokenizers[checkpoint] = AutoTokenizer.from_pretrained(checkpoint)
        self.tokenizer = self._tokenizers[checkpoint]

    def encode(self, texts):
        return self.tokenizer(list(texts), truncation=True, max_length=self.fixed["max_len"])["input_ids"]

    def lengths(self, encoded):
        return [len(e) for e in encoded]

    def collate(self, batch):
        pad_id = self.tokenizer.pad_token_id
        length = max(len(b[0]) for b in batch)
        ids = torch.full((len(batch), length), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), length), dtype=torch.long)
        for i, (s, _) in enumerate(batch):
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask[i, : len(s)] = 1
        return {"input_ids": ids, "attention_mask": mask}, torch.tensor([b[1] for b in batch], dtype=torch.long)

    def new_model(self) -> nn.Module:
        from transformers import AutoModelForSequenceClassification

        return HFClassifier(AutoModelForSequenceClassification.from_pretrained(self.checkpoint,
                                                                               num_labels=self.n_classes))

    def optimizer(self, model, n_steps: int):
        from transformers import get_linear_schedule_with_warmup

        no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
        groups = [
            {"params": [p for n, p in model.named_parameters() if not n.endswith(no_decay)],
             "weight_decay": self.params["weight_decay"]},
            {"params": [p for n, p in model.named_parameters() if n.endswith(no_decay)], "weight_decay": 0.0},
        ]
        opt = torch.optim.AdamW(groups, lr=self.params["lr"])
        sched = get_linear_schedule_with_warmup(opt, int(self.params["warmup_ratio"] * n_steps), n_steps)
        return opt, sched

    def save(self, model, path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        model.hf.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


# ---------------------------------------------------------------- loop

def predict_proba(model, setup, encoded, batch_size: int, device, fp16: bool) -> np.ndarray:
    """Softmax probabilities; batches are formed over length-sorted inputs, order restored."""
    order = np.argsort(setup.lengths(encoded), kind="stable")
    probs = np.zeros((len(encoded), setup.n_classes), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            inputs, _ = setup.collate([(encoded[i], 0) for i in idx])
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16 and device.type == "cuda"):
                logits = model(**inputs)
            probs[idx] = torch.softmax(logits.float(), dim=1).cpu().numpy()
    return probs


def train(setup, train_enc, ytr, val_enc, yva, fixed: dict, batch_size: int, seed: int, device, fp16: bool,
          grad_clip: float | None = None) -> dict:
    """Train with early stopping on val macro F1. Returns model (best epoch restored) and stats."""
    labels = list(range(setup.n_classes))
    model = setup.new_model().to(device)
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = DataLoader(list(zip(train_enc, ytr.tolist())), batch_size=batch_size, shuffle=True, generator=gen,
                        collate_fn=setup.collate, num_workers=0)
    opt, sched = setup.optimizer(model, fixed["max_epochs"] * len(loader))
    scaler = torch.amp.GradScaler("cuda", enabled=fp16 and device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss()
    best_f1, best_state, best_epoch, wait = -1.0, None, -1, 0
    history = []
    for epoch in range(fixed["max_epochs"]):
        t0 = time.perf_counter()
        model.train()
        for inputs, y in loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            y = y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16 and device.type == "cuda"):
                logits = model(**inputs)
            loss = loss_fn(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            if sched is not None:
                sched.step()
        t_train = time.perf_counter() - t0
        t1 = time.perf_counter()
        val_pred = predict_proba(model, setup, val_enc, fixed["eval_batch_size"], device, fp16).argmax(axis=1)
        f1 = f1_score(yva, val_pred, labels=labels, average="macro", zero_division=0)
        history.append({"epoch": epoch, "val_macro_f1": float(f1), "train_seconds": t_train,
                        "val_seconds": time.perf_counter() - t1})
        if f1 > best_f1:
            best_f1, best_epoch, wait = f1, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            wait += 1
            if wait >= fixed["patience"]:
                break
    model.load_state_dict(best_state)
    return {"model": model, "best_val_macro_f1": float(best_f1), "best_epoch": best_epoch,
            "epochs_run": len(history), "history": history}
