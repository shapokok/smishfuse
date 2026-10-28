"""Few-shot classification with an instruction LLM (no fine-tuning, no tuning).

Per eval seed: n_shots examples drawn from train, balanced over classes; the
prompt (configs/llm_prompt.yaml) is fixed. Greedy decoding of a short answer.
  * prediction: the answer word mapped to its class; any other answer -> "invalid"
    (counted as an error and reported as the invalid rate);
  * probabilities: softmax over the logits of the first token of each answer
    word at the first generated position (used for PR-AUC / ECE only).
Progress is checkpointed every `checkpoint_every_batches` batches.
"""
from __future__ import annotations

import re
import time

import numpy as np
import pandas as pd
import torch

from neural.common import Deferred, Timer, load_yaml, raw_path, save_raw, time_left

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
                                                 torch_dtype=getattr(torch, fixed["compute_dtype"]))
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


def run_llm_seed(task_name: str, task: dict, model_name: str, mcfg: dict, seed: int, train, test, log) -> dict:
    fixed = mcfg["fixed"]
    prompt = load_yaml(mcfg["prompt_file"])
    tcfg = prompt["tasks"][task_name]
    answers = tcfg["answers"]                       # answer word -> class name
    classes = task["classes"]
    word_of_class = {c: w for w, c in answers.items()}
    system = prompt["system"].format(label_description=tcfg["label_description"].strip(),
                                     answer_list=", ".join(answers))
    tr_ids, tr_texts, ytr = train
    te_ids, te_texts, yte = test

    with Timer() as t_load:
        tok, model = load_llm(mcfg["checkpoint"], fixed)
    first_ids = [tok.encode(word_of_class[c], add_special_tokens=False)[0] for c in classes]
    if len(set(first_ids)) != len(first_ids):
        raise RuntimeError(f"answer words share a first token: {dict(zip(classes, first_ids))}")

    shots = choose_shots(ytr, len(classes), fixed["n_shots"], seed)
    base = [{"role": "system", "content": system}]
    for i in shots:
        base.append({"role": "user", "content": prompt["user_template"].format(text=tr_texts[i])})
        base.append({"role": "assistant", "content": word_of_class[classes[ytr[i]]]})

    partial = raw_path(task_name, model_name, seed).with_suffix(".partial.csv")
    done = (pd.read_csv(partial, dtype={"id": str, "raw_answer": str}, keep_default_na=False)
            if partial.exists() else pd.DataFrame())
    done_ids = set(done["id"]) if len(done) else set()
    rows = done.to_dict("records") if len(done) else []
    order = [i for i in np.argsort([len(t) for t in te_texts], kind="stable") if te_ids[i] not in done_ids]
    bs = fixed["batch_size"]
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
        chats = [tok.apply_chat_template(base + [{"role": "user", "content": prompt["user_template"].format(
            text=te_texts[i])}], tokenize=False, add_generation_prompt=True) for i in idx]
        enc = tok(chats, return_tensors="pt", padding=True).to(model.device)
        n_prompt_tokens += int(enc["attention_mask"].sum())
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=fixed["max_new_tokens"], do_sample=False,
                                 output_scores=True, return_dict_in_generate=True, pad_token_id=tok.pad_token_id)
        first = out.scores[0][:, first_ids].float()
        probs = torch.softmax(first, dim=1).cpu().numpy()
        texts = tok.batch_decode(out.sequences[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for j, i in enumerate(idx):
            ans = normalize_answer(texts[j])
            row = {"id": te_ids[i], "raw_answer": texts[j], "valid": ans in answers,
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
    return {"seed": seed, "shot_train_ids": [tr_ids[i] for i in shots], "n_test": len(te_ids),
            "invalid_rate": float(np.mean([not r["valid"] for r in ordered])),
            "generation_seconds": gen_seconds, "model_load_seconds": t_load.seconds,
            "new_messages_this_session": len(order), "prompt_tokens_this_session": n_prompt_tokens}
