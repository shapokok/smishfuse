"""Training loop of the SmishFuse variants (shared by steps 4a, 4b and 4c).

fp16 autocast + GradScaler, AdamW with the encoder at the text lr and the new layers
at head_lr, linear schedule, gradient clipping, evaluation on val after every epoch,
early stopping on val macro F1; the best epoch's weights are restored at the end.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from fusion.model import SmishFuse
from neural.common import set_seed


class Batches:
    def __init__(self, enc, F, y, tokenizer):
        self.enc, self.F, self.y, self.tok = enc, F, y, tokenizer

    def collate(self, idx):
        out = {"features": torch.tensor(self.F[idx], dtype=torch.float32)}
        if self.enc is not None:
            seqs = [self.enc[i] for i in idx]
            length = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), length), self.tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(seqs), length), dtype=torch.long)
            for j, s in enumerate(seqs):
                ids[j, : len(s)] = torch.tensor(s, dtype=torch.long)
                mask[j, : len(s)] = 1
            out.update({"input_ids": ids, "attention_mask": mask})
        return out, torch.tensor(self.y[idx], dtype=torch.long)


def encode(tok, texts, max_len: int):
    return tok(list(texts), truncation=True, max_length=max_len)["input_ids"] if tok is not None else None


def predict(model, batches: Batches, device, bs: int, fp16: bool):
    n = len(batches.y)
    order = np.argsort([len(e) for e in batches.enc], kind="stable") if batches.enc is not None else np.arange(n)
    probs = np.zeros((n, model.head3.out_features))
    gate = np.full(n, np.nan)
    model.eval()
    with torch.no_grad():
        for s in range(0, n, bs):
            idx = order[s : s + bs]
            inputs, _ = batches.collate(idx)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16 and device.type == "cuda"):
                l3, _, g = model(**inputs)
            probs[idx] = torch.softmax(l3.float(), dim=1).cpu().numpy()
            if g is not None:
                gate[idx] = g.float().mean(dim=1).cpu().numpy()
    return probs, gate


def predict_texts(model, tok, texts, F, device, cfg):
    """Probabilities and gate means for an extra evaluation set (e.g. obfuscated val)."""
    b = Batches(encode(tok, texts, cfg["text"]["max_len"]), F, np.zeros(len(F), dtype=np.int64), tok)
    return predict(model, b, device, cfg["text"]["eval_batch_size"], True)


def fit(mode: str, cfg: dict, train, val, n_classes: int, seed: int, device, checkpoint: str,
        ham_index: int | None, model=None, tok=None, train_epoch=None) -> dict:
    """Train one variant. train / val = (texts, y, standardized features).

    ham_index: index of `ham` for the auxiliary binary head (ham vs malicious); None for a
    binary task, where the auxiliary head would repeat the main one and is not trained.
    model / tok: continue training an existing model (fine-tuning); otherwise a new one is built.
    train_epoch: optional callable epoch -> (texts, features) giving the training inputs of that
    epoch (augmentation); labels and order stay those of `train`. It must not use torch RNG.
    """
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    tc, fc = cfg["text"], cfg["fusion"]
    set_seed(seed, True)
    (tr_texts, ytr, Ftr), (va_texts, yva, Fva) = train, val
    if mode != "features" and tok is None:
        tok = AutoTokenizer.from_pretrained(checkpoint)
    train_b = Batches(encode(tok, tr_texts, tc["max_len"]), Ftr, ytr, tok)
    val_b = Batches(encode(tok, va_texts, tc["max_len"]), Fva, yva, tok)
    if model is None:
        model = SmishFuse(mode, Ftr.shape[1], fc["d"], fc["feature_mlp_hidden"], fc["dropout"], n_classes,
                          checkpoint).to(device)
    if mode == "features":
        fo = cfg["features_only"]
        bs, epochs, patience = fo["batch_size"], fo["max_epochs"], fo["patience"]
        opt, sched = torch.optim.Adam(model.parameters(), lr=fo["lr"]), None
    else:
        bs, epochs, patience = tc["batch_size"], tc["max_epochs"], tc["patience"]
        no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
        enc = model.encoder_parameters()
        enc_ids = {id(p) for _, p in enc}
        groups = [
            {"params": [p for n, p in enc if not n.endswith(no_decay)], "lr": tc["lr"],
             "weight_decay": tc["weight_decay"]},
            {"params": [p for n, p in enc if n.endswith(no_decay)], "lr": tc["lr"], "weight_decay": 0.0},
            {"params": [p for p in model.parameters() if id(p) not in enc_ids], "lr": fc["head_lr"],
             "weight_decay": 0.0},
        ]
        opt = torch.optim.AdamW(groups)
        n_steps = epochs * int(np.ceil(len(ytr) / bs))
        sched = get_linear_schedule_with_warmup(opt, int(tc["warmup_ratio"] * n_steps), n_steps)
    gen = torch.Generator()
    gen.manual_seed(seed)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ce = nn.CrossEntropyLoss()
    best = {"f1": -1.0}
    wait, history = 0, []
    t_start = time.perf_counter()
    for epoch in range(epochs):
        t0 = time.perf_counter()
        if train_epoch is not None:
            ep_texts, ep_F = train_epoch(epoch)
            train_b = Batches(encode(tok, ep_texts, tc["max_len"]), ep_F, ytr, tok)
        model.train()
        perm = torch.randperm(len(ytr), generator=gen).numpy()
        for s in range(0, len(perm), bs):
            inputs, y = train_b.collate(perm[s : s + bs])
            inputs = {k: v.to(device) for k, v in inputs.items()}
            y = y.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                l3, lb, _ = model(**inputs)
            loss = ce(l3.float(), y)
            if ham_index is not None:
                loss = loss + fc["binary_loss_weight"] * ce(lb.float(), (y != ham_index).long())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if mode != "features":
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            scaler.step(opt)
            scaler.update()
            if sched is not None:
                sched.step()
        probs, gate = predict(model, val_b, device, tc["eval_batch_size"], True)
        f1 = f1_score(yva, probs.argmax(axis=1), labels=list(range(n_classes)), average="macro", zero_division=0)
        history.append({"epoch": epoch, "val_macro_f1": float(f1), "seconds": time.perf_counter() - t0})
        if f1 > best["f1"]:
            best = {"f1": f1, "epoch": epoch, "probs": probs, "gate": gate,
                    "state": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best["state"])
    return {"model": model, "tok": tok, "best_val_macro_f1": float(best["f1"]), "best_epoch": best["epoch"],
            "epochs_run": len(history), "history": history, "seconds": time.perf_counter() - t_start,
            "batch_size": bs, "val_probs": best["probs"], "val_gate": best["gate"]}
