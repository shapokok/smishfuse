"""Step 5a: re-implemented published methods and the classical models of the inflation CV.

Every fit_* function trains on (texts, y) and returns an object with predict_proba(texts) -> (n, k).
Settings come from configs/step5a.yaml; what the papers do not state is marked "ours" there.

* mahmud2024_cnn_bigru: Word2Vec (gensim) + Conv1D + max pooling + BiGRU + sigmoid output
  (Mahmud et al., Systems 2024). Written in PyTorch to mirror the paper's Keras model: Keras
  initializers, Keras GRU (reset_after) with one input and one recurrent dropout mask per sequence,
  Adam with the Keras epsilon, "pre" padding and truncation, unknown words dropped (Keras Tokenizer).
* shim2024_*: fine-tuned transformers with the paper's grid over batch size and learning rate
  (Shim et al., arXiv:2411.02403); the choice is made on an inner split of our train.
* linear_svm / logreg: step-2 pipelines with the best mendeley_3class parameters.
"""
from __future__ import annotations

import itertools
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from neural.common import read_json, set_seed

ROOT = Path(__file__).resolve().parents[2]
CFG2 = yaml.safe_load((ROOT / "configs" / "step2.yaml").read_text(encoding="utf-8"))
N_JOBS = 2  # two workers share the 4 CPU cores of the Kaggle GPU machine


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------- classical (inflation CV)

def fit_classic(model_name: str, texts, y, seed: int):
    import classic

    best = read_json(ROOT / "results" / "best_params" / "mendeley_3class" / f"{model_name}.json")
    model, bw = classic.build_model(model_name, best["model_params"], CFG2["models"][model_name]["fixed"], seed, N_JOBS)
    clf = classic.TextClassifier(classic.build_vectorizer(best["vectorizer"], CFG2["vectorizer_fixed"]), model, bw)
    texts, y = np.asarray(texts, dtype=object), np.asarray(y)
    perm = np.random.default_rng(seed).permutation(len(y))  # train order of the step-2 final runs
    return clf.fit(texts[perm], y[perm])


# ---------------------------------------------------------------- Mahmud et al. 2024

_NON_ALNUM = re.compile(r"[^a-z0-9\s]")


def mahmud_tokens(text: str, stop: frozenset) -> list[str]:
    """Lowercase, punctuation and special characters removed, whitespace tokens, stop words removed."""
    return [w for w in _NON_ALNUM.sub(" ", text.lower()).split() if w not in stop]


class KerasGRU(nn.Module):
    """Keras GRU (reset_after=True, tanh / sigmoid; glorot_uniform kernel, orthogonal recurrent
    kernel, zero biases). Dropout as in Keras: one input mask and one recurrent mask per sequence,
    only in training. Returns the last hidden state."""

    def __init__(self, n_in: int, units: int, dropout: float, recurrent_dropout: float):
        super().__init__()
        self.units, self.dropout, self.recurrent_dropout = units, dropout, recurrent_dropout
        self.kernel = nn.Parameter(torch.empty(n_in, 3 * units))
        self.recurrent = nn.Parameter(torch.empty(units, 3 * units))
        self.bias = nn.Parameter(torch.zeros(2, 3 * units))  # input and recurrent biases (reset_after)
        nn.init.xavier_uniform_(self.kernel)
        nn.init.orthogonal_(self.recurrent)

    def _mask(self, ref: torch.Tensor, shape, rate: float):
        if not self.training or rate <= 0:
            return None
        return torch.bernoulli(ref.new_full(shape, 1 - rate)) / (1 - rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (batch, time, n_in)
        b, t, _ = x.shape
        u = self.units
        dm = self._mask(x, (b, 1, x.shape[2]), self.dropout)
        if dm is not None:
            x = x * dm
        xp = x @ self.kernel + self.bias[0]
        rm = self._mask(x, (b, u), self.recurrent_dropout)
        h = x.new_zeros(b, u)
        for i in range(t):
            hp = (h if rm is None else h * rm) @ self.recurrent + self.bias[1]
            xz, xr, xh = xp[:, i].split(u, dim=1)
            hz, hr, hh = hp.split(u, dim=1)
            z = torch.sigmoid(xz + hz)
            r = torch.sigmoid(xr + hr)
            h = z * h + (1 - z) * torch.tanh(xh + r * hh)
        return h


class CNNBiGRU(nn.Module):
    def __init__(self, emb: np.ndarray, mc: dict):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(torch.tensor(emb, dtype=torch.float32),
                                                freeze=not mc["embedding_trainable"], padding_idx=None)
        self.conv = nn.Conv1d(emb.shape[1], mc["conv"]["filters"], mc["conv"]["kernel"])
        self.pool = nn.MaxPool1d(mc["pool"])
        g = mc["gru"]
        self.fwd = KerasGRU(mc["conv"]["filters"], g["units"], g["dropout"], g["recurrent_dropout"])
        self.bwd = KerasGRU(mc["conv"]["filters"], g["units"], g["dropout"], g["recurrent_dropout"])
        self.out = nn.Linear(2 * g["units"], 1)
        for layer in (self.conv, self.out):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.relu(self.conv(self.emb(ids).transpose(1, 2)))).transpose(1, 2)
        return self.out(torch.cat([self.fwd(x), self.bwd(x.flip(1))], dim=1)).squeeze(1)


class MahmudModel:
    def __init__(self, mc: dict, device):
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        self.mc, self.device = mc, device
        self.stop = frozenset(ENGLISH_STOP_WORDS)

    def encode(self, texts) -> torch.Tensor:
        n = self.mc["max_len"]
        out = np.zeros((len(texts), n), dtype=np.int64)
        for i, t in enumerate(texts):
            seq = [self.vocab[w] for w in mahmud_tokens(t, self.stop) if w in self.vocab][-n:]  # truncating "pre"
            if seq:
                out[i, n - len(seq):] = seq  # padding "pre"
        return torch.tensor(out)

    def fit(self, texts, y, seed: int) -> "MahmudModel":
        from gensim.models import Word2Vec

        mc = self.mc
        set_seed(seed, True)
        sents = [mahmud_tokens(t, self.stop) for t in texts]
        w2v = Word2Vec([s for s in sents if s], vector_size=mc["word2vec"]["vector_size"],
                       min_count=mc["word2vec"]["min_count"], seed=seed, workers=1)
        words = list(w2v.wv.index_to_key)
        self.vocab = {w: i + 1 for i, w in enumerate(words)}  # 0 = padding
        emb = np.zeros((len(words) + 1, mc["word2vec"]["vector_size"]), dtype=np.float32)
        emb[1:] = w2v.wv.vectors
        self.model = CNNBiGRU(emb, mc).to(self.device)
        X = self.encode(texts)
        yt = torch.tensor(np.asarray(y), dtype=torch.float32)
        opt = torch.optim.Adam(self.model.parameters(), lr=mc["lr"], eps=1e-7)  # Keras Adam epsilon
        loss_fn = nn.BCEWithLogitsLoss()
        gen = torch.Generator()
        gen.manual_seed(seed)
        bs = mc["batch_size"]
        for _ in range(mc["epochs"]):
            self.model.train()
            perm = torch.randperm(len(yt), generator=gen)
            for s in range(0, len(perm), bs):
                idx = perm[s : s + bs]
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(self.model(X[idx].to(self.device)), yt[idx].to(self.device))
                loss.backward()
                opt.step()
        return self

    def predict_proba(self, texts) -> np.ndarray:
        X = self.encode(texts)
        self.model.eval()
        ps = []
        with torch.no_grad():
            for s in range(0, len(X), 512):
                ps.append(torch.sigmoid(self.model(X[s : s + 512].to(self.device))).cpu().numpy())
        p = np.concatenate(ps).astype(np.float64)
        return np.stack([1 - p, p], axis=1)


def fit_mahmud(mc: dict, texts, y, seed: int, device):
    return MahmudModel(mc, device).fit(list(texts), y, seed)


# ---------------------------------------------------------------- Shim et al. 2024

class Transformer:
    def __init__(self, checkpoint: str, max_len: int, device):
        from transformers import AutoTokenizer

        self.checkpoint, self.max_len, self.device = checkpoint, max_len, device
        self.tok = AutoTokenizer.from_pretrained(checkpoint)

    def _enc(self, texts):
        return self.tok(list(texts), truncation=True, max_length=self.max_len)["input_ids"]

    def _batch(self, seqs):
        n = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), n), self.tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), n), dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask[j, : len(s)] = 1
        return ids.to(self.device), mask.to(self.device)

    def train(self, texts, y, bs: int, lr: float, epochs: int, seed: int, on_epoch=None) -> "Transformer":
        """Constant lr, AdamW (torch defaults otherwise), fp16 autocast; on_epoch(epoch) after every epoch."""
        from transformers import AutoModelForSequenceClassification

        set_seed(seed, True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint, num_labels=2).to(self.device)
        enc = self._enc(texts)
        yt = torch.tensor(np.asarray(y), dtype=torch.long)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr)
        scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")
        gen = torch.Generator()
        gen.manual_seed(seed)
        for epoch in range(epochs):
            self.model.train()
            perm = torch.randperm(len(yt), generator=gen).tolist()
            for s in range(0, len(perm), bs):
                idx = perm[s : s + bs]
                ids, mask = self._batch([enc[i] for i in idx])
                with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                                    enabled=self.device.type == "cuda"):
                    logits = self.model(input_ids=ids, attention_mask=mask).logits
                loss = nn.functional.cross_entropy(logits.float(), yt[idx].to(self.device))
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            if on_epoch is not None:
                on_epoch(epoch)
        return self

    def predict_proba(self, texts) -> np.ndarray:
        enc = self._enc(texts)
        order = np.argsort([len(e) for e in enc], kind="stable")
        probs = np.zeros((len(enc), 2))
        self.model.eval()
        with torch.no_grad():
            for s in range(0, len(order), 128):
                idx = order[s : s + 128]
                ids, mask = self._batch([enc[i] for i in idx])
                with torch.autocast(device_type=self.device.type, dtype=torch.float16,
                                    enabled=self.device.type == "cuda"):
                    logits = self.model(input_ids=ids, attention_mask=mask).logits
                probs[idx] = torch.softmax(logits.float(), dim=1).cpu().numpy()
        return probs

    def release(self) -> None:
        del self.model
        torch.cuda.empty_cache()


def macro_f1(y, pred) -> float:
    return float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0))


def select_shim(sc: dict, checkpoint: str, texts, y, seed: int, device) -> dict:
    """The paper's grid (batch size x lr, up to max_epochs epochs) on a stratified inner split of
    train; best inner-val macro F1, ties -> the first in (batch, lr, epoch) order."""
    texts, y = np.asarray(texts, dtype=object), np.asarray(y)
    tr, va = train_test_split(np.arange(len(y)), test_size=sc["inner_val"], stratify=y, random_state=seed)
    history, best = [], None
    for bs, lr in itertools.product(sc["batch_sizes"], sc["lrs"]):
        m = Transformer(checkpoint, sc["max_len"], device)

        def on_epoch(epoch, m=m, bs=bs, lr=lr):
            nonlocal best
            f1 = macro_f1(y[va], m.predict_proba(texts[va]).argmax(axis=1))
            history.append({"batch_size": bs, "lr": lr, "epoch": epoch, "inner_val_macro_f1": f1})
            if best is None or f1 > best["inner_val_macro_f1"]:
                best = history[-1]

        m.train(texts[tr], y[tr], bs, lr, sc["max_epochs"], seed, on_epoch)
        m.release()
        log(f"[shim] {checkpoint} bs {bs} lr {lr}: best so far {best}")
    return {"chosen": best, "history": history, "n_inner_train": len(tr), "n_inner_val": len(va)}


def fit_shim(sc: dict, checkpoint: str, texts, y, seed: int, device, choice: dict):
    """Retrain on the whole train with the chosen batch size, lr and number of epochs."""
    return Transformer(checkpoint, sc["max_len"], device).train(
        list(texts), y, choice["batch_size"], choice["lr"], choice["epoch"] + 1, seed)
