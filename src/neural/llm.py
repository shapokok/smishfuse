"""Few-shot classification with an instruction LLM (no fine-tuning, no tuning).

Per eval seed: n_shots examples drawn from train, balanced over classes; the
prompt (configs/llm_prompt.yaml) is fixed. Greedy decoding of a short answer.
  * prediction: the answer word mapped to its class; any other answer -> "invalid"
    (counted as an error and reported as the invalid rate);
  * probabilities: softmax over the logits of the first token of each answer
    word at the first generated position (used for PR-AUC / ECE only).
Progress is checkpointed every `checkpoint_every_batches` batches.

Prefix KV cache (fixed.prefix_cache): the system message and the shots are the
same for every message of a seed, so their key/value cache is computed once and
only the per-message suffix is run through the model. Batches are laid out as
[prefix][left padding][suffix] with the padding masked; positions come from the
attention mask, so they match the uncached layout [left padding][prefix][suffix].
`cache_equivalence_check` compares both modes on val messages.
"""
from __future__ import annotations

import copy
import re
import time

import numpy as np
import pandas as pd
import torch

from neural.common import RESULTS, Deferred, Timer, load_yaml, raw_path, save_raw, time_left, write_json

_MODEL_CACHE: dict = {}


def load_llm(checkpoint: str, fixed: dict):
    if checkpoint in _MODEL_CACHE:
        return _MODEL_CACHE[checkpoint]
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(checkpoint)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    quant = None
    if fixed["load_in_4bit"]:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=fixed["bnb_4bit_quant_type"],
                                   bnb_4bit_compute_dtype=getattr(torch, fixed["compute_dtype"]))
    model = AutoModelForCausalLM.from_pretrained(checkpoint, quantization_config=quant, device_map={"": 0},
                                                 dtype=getattr(torch, fixed["compute_dtype"]))
    model.eval()
    _MODEL_CACHE[checkpoint] = (tok, model)
    return tok, model


def choose_shots(y_train: np.ndarray, n_classes: int, n_shots: int, seed: int) -> list[int]:
    """Indices into train: n_shots // K per class, remainder to seed-chosen classes; shuffled."""
    rng = np.random.default_rng(seed)
    per = np.full(n_classes, n_shots // n_classes)
    rem = n_shots - per.sum()
    if rem:
        per[rng.choice(n_classes, size=rem, replace=False)] += 1
    idx = []
    for c in range(n_classes):
        pool = np.flatnonzero(y_train == c)
        idx += rng.choice(pool, size=per[c], replace=False).tolist()
    return [idx[i] for i in rng.permutation(len(idx))]


def normalize_answer(text: str) -> str:
    return re.sub(r"^[^\w]+|[^\w]+$", "", text.strip().lower())


class FewShotClassifier:
    """Prompt construction and batched greedy answers for one (task, seed)."""

    def __init__(self, task_name: str, task: dict, mcfg: dict, train, seed: int):
        self.fixed = mcfg["fixed"]
        prompt = load_yaml(mcfg["prompt_file"])
        tcfg = prompt["tasks"][task_name]
        self.answers = tcfg["answers"]  # answer word -> class name
        self.classes = task["classes"]
        word_of_class = {c: w for w, c in self.answers.items()}
        self.user_template = prompt["user_template"]
        system = prompt["system"].format(label_description=tcfg["label_description"].strip(),
                                         answer_list=", ".join(self.answers))
        tr_ids, tr_texts, ytr = train
        self.shots = choose_shots(ytr, len(self.classes), self.fixed["n_shots"], seed)
        self.shot_ids = [tr_ids[i] for i in self.shots]
        self.base = [{"role": "system", "content": system}]
        for i in self.shots:
            self.base.append({"role": "user", "content": self.user_template.format(text=tr_texts[i])})
            self.base.append({"role": "assistant", "content": word_of_class[self.classes[ytr[i]]]})
        with Timer() as t:
            self.tok, self.model = load_llm(mcfg["checkpoint"], self.fixed)
        self.load_seconds = t.seconds
        self.first_ids = [self.tok.encode(word_of_class[c], add_special_tokens=False)[0] for c in self.classes]
        if len(set(self.first_ids)) != len(self.first_ids):
            raise RuntimeError(f"answer words share a first token: {dict(zip(self.classes, self.first_ids))}")
        self.prefix_text = self.tok.apply_chat_template(self.base, tokenize=False, add_generation_prompt=False)
        self.prefix_ids = self.tok(self.prefix_text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        self._prefix_cache = {}  # batch size -> KV cache of the prefix repeated batch-size times

    def chat(self, text: str) -> str:
        msgs = self.base + [{"role": "user", "content": self.user_template.format(text=text)}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def _decode(self, out, n_input: int):
        probs = torch.softmax(out.scores[0][:, self.first_ids].float(), dim=1).cpu().numpy()
        texts = self.tok.batch_decode(out.sequences[:, n_input:], skip_special_tokens=True)
        return texts, probs

    def answer(self, texts: list[str], use_cache: bool):
        """Returns (raw answers, class probabilities, number of prompt tokens processed)."""
        gen = dict(max_new_tokens=self.fixed["max_new_tokens"], do_sample=False, output_scores=True,
                   return_dict_in_generate=True, pad_token_id=self.tok.pad_token_id)
        if not use_cache:
            enc = self.tok([self.chat(t) for t in texts], return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(**enc, **gen)
            raw, probs = self._decode(out, enc["input_ids"].shape[1])
            return raw, probs, int(enc["attention_mask"].sum())

        suffixes = []
        for t in texts:
            full = self.chat(t)
            if not full.startswith(self.prefix_text):
                raise RuntimeError("chat template output does not start with the shared prefix")
            ids = self.tok(full[len(self.prefix_text):], add_special_tokens=False)["input_ids"]
            whole = self.tok(full, add_special_tokens=False)["input_ids"]
            if whole != self.prefix_ids[0].tolist() + ids:
                raise RuntimeError("prefix + suffix tokenization differs from tokenizing the whole prompt")
            suffixes.append(ids)
        b, p = len(texts), self.prefix_ids.shape[1]
        s = max(len(x) for x in suffixes)
        suf = torch.full((b, s), self.tok.pad_token_id, dtype=torch.long)
        suf_mask = torch.zeros((b, s), dtype=torch.long)
        for i, x in enumerate(suffixes):
            suf[i, s - len(x):] = torch.tensor(x)
            suf_mask[i, s - len(x):] = 1
        device = self.model.device
        input_ids = torch.cat([self.prefix_ids.repeat(b, 1), suf], dim=1).to(device)
        attention = torch.cat([torch.ones((b, p), dtype=torch.long), suf_mask], dim=1).to(device)
        if b not in self._prefix_cache:
            with torch.no_grad():
                # Only the KV cache is needed: logits_to_keep=1 avoids materialising
                # batch x prefix_len x vocab logits (4-6 GB, OOM on the T4 in the first full run).
                self._prefix_cache[b] = self.model(self.prefix_ids.repeat(b, 1).to(device), use_cache=True,
                                                   logits_to_keep=1).past_key_values
        past = copy.deepcopy(self._prefix_cache[b])  # generate() extends the cache in place
        with torch.no_grad():
            out = self.model.generate(input_ids=input_ids, attention_mask=attention, past_key_values=past, **gen)
        raw, probs = self._decode(out, input_ids.shape[1])
        return raw, probs, int(suf_mask.sum())


def run_llm_seed(task_name: str, task: dict, model_name: str, mcfg: dict, seed: int, train, test, log) -> dict:
    fixed = mcfg["fixed"]
    clf = FewShotClassifier(task_name, task, mcfg, train, seed)
    classes, answers = clf.classes, clf.answers
    te_ids, te_texts, yte = test
    use_cache = bool(fixed["prefix_cache"])
    bs = fixed["cache_batch_size"] if use_cache else fixed["batch_size"]

    partial = raw_path(task_name, model_name, seed).with_suffix(".partial.csv")
    done = (pd.read_csv(partial, dtype={"id": str, "raw_answer": str}, keep_default_na=False)
            if partial.exists() else pd.DataFrame())
    done_ids = set(done["id"]) if len(done) else set()
    rows = done.to_dict("records") if len(done) else []
    if done_ids:
        log(f"[{task_name}/{model_name}] seed {seed}: resuming, {len(done_ids)} messages already answered")
    order = [i for i in np.argsort([len(t) for t in te_texts], kind="stable") if te_ids[i] not in done_ids]
    n_prompt_tokens = 0
    t0 = time.perf_counter()
    batch_seconds = 0.0
    for b, start in enumerate(range(0, len(order), bs)):
        if time_left() < 3 * batch_seconds + 60:
            partial.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(partial, index=False, lineterminator="\n")
            raise Deferred(f"{task_name}/{model_name} seed {seed}: {len(rows)}/{len(te_ids)} messages, deadline")
        t_batch = time.perf_counter()
        idx = order[start : start + bs]
        raw, probs, n_tok = clf.answer([te_texts[i] for i in idx], use_cache)
        n_prompt_tokens += n_tok
        for j, i in enumerate(idx):
            ans = normalize_answer(raw[j])
            row = {"id": te_ids[i], "raw_answer": raw[j], "valid": ans in answers,
                   "y_pred": answers.get(ans, "invalid")}
            for k, c in enumerate(classes):
                row[f"prob_{c}"] = float(probs[j, k])
            rows.append(row)
        batch_seconds = max(batch_seconds, time.perf_counter() - t_batch)
        if (b + 1) % fixed["checkpoint_every_batches"] == 0:
            partial.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(partial, index=False, lineterminator="\n")
            log(f"  [{task_name}/{model_name}] seed {seed}: {len(rows)}/{len(te_ids)} messages")
    gen_seconds = time.perf_counter() - t0

    by_id = {r["id"]: r for r in rows}
    ordered = [by_id[i] for i in te_ids]
    probs = np.array([[r[f"prob_{c}"] for c in classes] for r in ordered])
    save_raw(task_name, model_name, seed, te_ids, yte, [r["y_pred"] for r in ordered], probs, classes,
             extra={"valid": [r["valid"] for r in ordered], "raw_answer": [r["raw_answer"] for r in ordered]})
    if partial.exists():
        partial.unlink()
    return {"seed": seed, "shot_train_ids": clf.shot_ids, "n_test": len(te_ids),
            "invalid_rate": float(np.mean([not r["valid"] for r in ordered])), "prefix_cache": use_cache,
            "batch_size": bs, "generation_seconds": gen_seconds, "model_load_seconds": clf.load_seconds,
            "new_messages_this_session": len(order), "prompt_tokens_this_session": n_prompt_tokens}


def cache_equivalence_check(task_name: str, task: dict, model_name: str, mcfg: dict, train, val, n: int,
                            seed: int, log) -> dict:
    """Answers with and without the prefix cache on n val messages (random sample, seed 0)."""
    fixed = mcfg["fixed"]
    clf = FewShotClassifier(task_name, task, mcfg, train, seed)
    va_ids, va_texts, _ = val
    pick = sorted(np.random.default_rng(0).choice(len(va_ids), size=min(n, len(va_ids)), replace=False).tolist())
    texts = [va_texts[i] for i in pick]
    result = {}
    for mode, use_cache, bs in [("no_cache", False, fixed["batch_size"]), ("cache", True, fixed["cache_batch_size"])]:
        raw, probs = [], []
        with Timer() as t:
            for start in range(0, len(texts), bs):
                r, p, _ = clf.answer(texts[start : start + bs], use_cache)
                raw += r
                probs.append(p)
        result[mode] = {"raw": raw, "probs": np.concatenate(probs), "seconds": t.seconds, "batch_size": bs}
    same_raw = [a == b for a, b in zip(result["no_cache"]["raw"], result["cache"]["raw"])]
    same_norm = [normalize_answer(a) == normalize_answer(b)
                 for a, b in zip(result["no_cache"]["raw"], result["cache"]["raw"])]
    out = {
        "task": task_name, "seed_of_shots": seed, "n_messages": len(texts),
        "identical_raw_answers": int(sum(same_raw)), "identical_normalized_answers": int(sum(same_norm)),
        "all_identical": bool(all(same_raw)),
        "max_abs_prob_diff": float(np.abs(result["no_cache"]["probs"] - result["cache"]["probs"]).max()),
        "seconds_no_cache": result["no_cache"]["seconds"], "seconds_cache": result["cache"]["seconds"],
        "batch_size_no_cache": result["no_cache"]["batch_size"], "batch_size_cache": result["cache"]["batch_size"],
        "mismatches": [{"id": va_ids[pick[i]], "no_cache": result["no_cache"]["raw"][i],
                        "cache": result["cache"]["raw"][i]} for i, ok in enumerate(same_raw) if not ok],
    }
    write_json(out, RESULTS / "llm_cache_check" / f"{task_name}.json")
    log(f"[{task_name}/{model_name}] cache check: {out['identical_raw_answers']}/{len(texts)} identical answers, "
        f"max |dp| {out['max_abs_prob_diff']:.2e}, {out['seconds_no_cache']:.0f}s -> {out['seconds_cache']:.0f}s")
    return out
