"""Step-3 worker: runs a queue of (task, model) jobs on one GPU.

For trainable models: Optuna tuning on val (one JSON per finished trial), best
parameters, then the eval seeds on test (one raw CSV + one JSON per seed, model
weights for save_model_seed). For the LLM: the eval seeds only.
Finished trials / seeds found in the results folder are skipped, so a restarted
run continues where the previous one stopped. A failing job is logged to
results/errors/ and the worker moves on to the next job.

Run: python src/neural/run.py --mode smoke|full --jobs '[{"task": "moz_binary", "model": "textcnn"}]'
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # must be set before CUDA initialises

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import optuna  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from eval.metrics import compute_metrics  # noqa: E402
from neural.common import (RESULTS, Deferred, Timer, distributions, load_configs, load_split,  # noqa: E402
                           raw_path, read_json, save_raw, set_seed, suggest, time_left, write_json)

LOG = None


def log(msg: str) -> None:
    print(msg, flush=True)
    if LOG is not None:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def make_setup(model_name: str, mcfg: dict, params: dict, task_name: str, train_texts, n_classes: int):
    from neural.trainers import TransformerSetup, WordSetup

    if mcfg["kind"] == "textnn":
        return WordSetup(model_name, params, train_texts, n_classes)
    ckpt = mcfg.get("checkpoint_by_task", {}).get(task_name, mcfg["checkpoint"])
    return TransformerSetup(ckpt, params, mcfg["fixed"], n_classes)


def fit_once(model_name, mcfg, params, task_name, data, n_classes, seed, proto, device):
    from neural.trainers import train

    set_seed(seed, proto["deterministic"])
    _, tr_texts, ytr = data["train"]
    setup = make_setup(model_name, mcfg, params, task_name, tr_texts, n_classes)
    with Timer() as t_enc:
        tr_enc = setup.encode(tr_texts)
        va_enc = setup.encode(data["val"][1])
    res = train(setup, tr_enc, ytr, va_enc, data["val"][2], mcfg["fixed"], params["batch_size"], seed, device,
                proto["fp16"], grad_clip=mcfg["fixed"].get("grad_clip"))
    res["encode_seconds"] = t_enc.seconds
    res["setup"] = setup
    return res


def tune(task_name, model_name, mcfg, data, n_classes, proto, n_trials, device) -> dict:
    tdir = RESULTS / "tuning" / task_name / model_name
    done = sorted(tdir.glob("trial_*.json"))
    space = mcfg["space"]
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=proto["tuning_seed"] + len(done)))
    dists = distributions(space)
    for p in done:
        t = read_json(p)
        study.add_trial(optuna.trial.create_trial(params=t["params"], distributions=dists, value=t["value"]))
    if done:
        log(f"[{task_name}/{model_name}] resuming tuning: {len(done)} trials already done")

    def objective(trial):
        number = trial.number  # counts the trials re-added from disk
        params = suggest(trial, space)
        with Timer() as t:
            res = fit_once(model_name, mcfg, params, task_name, data, n_classes, proto["tuning_seed"], proto, device)
        write_json({"number": number, "params": params, "value": res["best_val_macro_f1"],
                    "best_epoch": res["best_epoch"], "epochs_run": res["epochs_run"], "history": res["history"],
                    "encode_seconds": res["encode_seconds"], "seconds": t.seconds,
                    "n_train": len(data["train"][2]), "n_val": len(data["val"][2]), "device": str(device)},
                   tdir / f"trial_{number:03d}.json")
        log(f"[{task_name}/{model_name}] trial {number}: val macro F1 {res['best_val_macro_f1']:.4f} "
            f"({res['epochs_run']} epochs, {t.seconds:.0f}s) {params}")
        value = res["best_val_macro_f1"]
        del res
        torch.cuda.empty_cache()
        return value

    durations = [read_json(p)["seconds"] for p in done]

    def stop_near_deadline(st, trial):
        durations.append(read_json(tdir / f"trial_{trial.number:03d}.json")["seconds"])
        if time_left() < 1.3 * max(durations):
            st.stop()

    remaining = n_trials - len(done)
    if remaining > 0 and done and time_left() < 1.3 * max(durations):
        raise Deferred(f"{task_name}/{model_name}: not enough time for another trial")
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, n_jobs=1, callbacks=[stop_near_deadline])
        if len(list(tdir.glob("trial_*.json"))) < n_trials:
            raise Deferred(f"{task_name}/{model_name}: tuning stopped at the session deadline")
    trials = sorted((read_json(p) for p in tdir.glob("trial_*.json")), key=lambda t: t["number"])[:n_trials]
    best = max(trials, key=lambda t: (t["value"], -t["number"]))
    out = {"task": task_name, "model": model_name, "trial": best["number"], "val_macro_f1": best["value"],
           "model_params": best["params"], "n_trials": len(trials),
           "checkpoint": mcfg.get("checkpoint_by_task", {}).get(task_name, mcfg.get("checkpoint"))}
    write_json(out, RESULTS / "best_params" / task_name / f"{model_name}.json")
    return out


def final_seed(task_name, task, model_name, mcfg, data, best, seed, proto, device):
    from neural.trainers import predict_proba

    classes = task["classes"]
    with Timer() as t:
        res = fit_once(model_name, mcfg, best["model_params"], task_name, data, len(classes), seed, proto, device)
    setup, model = res["setup"], res["model"]
    te_ids, te_texts, yte = data["test"]
    with Timer() as t_pred:
        probs = predict_proba(model, setup, setup.encode(te_texts), mcfg["fixed"]["eval_batch_size"], device,
                              proto["fp16"])
    pred = probs.argmax(axis=1)
    save_raw(task_name, model_name, seed, te_ids, yte, [classes[i] for i in pred], probs, classes)
    if seed == proto["save_model_seed"]:
        setup.save(model, RESULTS / "models" / task_name / model_name / f"seed{seed}")
    m = compute_metrics(yte, probs, classes, positive=task["positive"])
    write_json({"task": task_name, "model": model_name, "seed": seed, "metrics": m,
                "best_epoch": res["best_epoch"], "epochs_run": res["epochs_run"], "history": res["history"],
                "train_seconds": t.seconds, "test_predict_seconds": t_pred.seconds,
                "n_train": len(data["train"][2]), "n_test": len(yte), "device": str(device)},
               RESULTS / "seed_runs" / task_name / model_name / f"seed{seed}.json")
    log(f"[{task_name}/{model_name}] seed {seed}: test macro F1 {m['macro_f1']:.4f} ({t.seconds:.0f}s)")


def run_job(job: dict, mode: str, device, overrides: dict) -> None:
    from neural.llm import cache_equivalence_check, run_llm_seed

    step2, step3 = load_configs()
    proto = dict(step3["protocol"])
    fraction = None
    if mode == "smoke":
        proto.update({k: v for k, v in step3["smoke"].items() if k in ("n_trials", "eval_seeds")})
        fraction = step3["smoke"]["data_fraction"]
    proto.update(overrides)  # e.g. smoke test of resume: more trials than the default smoke
    task_name, model_name = job["task"], job["model"]
    task = step2["tasks"][task_name]
    mcfg = step3["models"][model_name]
    if job.get("check_cache"):
        # Always on full-size train/val: the check needs cache_check_messages val messages.
        out = RESULTS / "llm_cache_check" / f"{task_name}.json"
        if out.exists():
            log(f"[{task_name}/{model_name}] cache check already done, skipped")
            return
        cache_equivalence_check(task_name, task, model_name, mcfg, load_split(task, "train"),
                                load_split(task, "val"), mcfg["fixed"]["cache_check_messages"],
                                proto["eval_seeds"][0], log)
        return
    data = {s: load_split(task, s, fraction) for s in ["train", "val", "test"]}
    log(f"[{task_name}/{model_name}] mode {mode}: train {len(data['train'][2])}, val {len(data['val'][2])}, "
        f"test {len(data['test'][2])}")

    if "seeds" in job and mode == "full":  # a job may cover only some seeds (LLM split over two GPUs)
        proto["eval_seeds"] = job["seeds"]
    if mcfg["kind"] == "llm":
        for seed in proto["eval_seeds"]:
            if raw_path(task_name, model_name, seed).exists():
                log(f"[{task_name}/{model_name}] seed {seed}: already done, skipped")
                continue
            info = run_llm_seed(task_name, task, model_name, mcfg, seed, data["train"], data["test"], log)
            df_raw = pd.read_csv(raw_path(task_name, model_name, seed))
            idx = {c: i for i, c in enumerate(task["classes"])}
            y_pred = df_raw["y_pred"].map(lambda v: idx.get(v, -1)).to_numpy()
            probs = df_raw[[f"prob_{c}" for c in task["classes"]]].to_numpy()
            m = compute_metrics(data["test"][2], probs, task["classes"], positive=task["positive"], y_pred=y_pred)
            write_json({"task": task_name, "model": model_name, "metrics": m, **info,
                        "checkpoint": mcfg["checkpoint"], "device": str(device)},
                       RESULTS / "seed_runs" / task_name / model_name / f"seed{seed}.json")
            log(f"[{task_name}/{model_name}] seed {seed}: test macro F1 {m['macro_f1']:.4f}, invalid "
                f"{info['invalid_rate']:.3%}, {info['generation_seconds']:.0f}s")
        return

    best_path = RESULTS / "best_params" / task_name / f"{model_name}.json"
    n_done = len(list((RESULTS / "tuning" / task_name / model_name).glob("trial_*.json")))
    if best_path.exists() and n_done >= proto["n_trials"]:
        best = read_json(best_path)
    else:
        best = tune(task_name, model_name, mcfg, data, len(task["classes"]), proto, proto["n_trials"], device)
    log(f"[{task_name}/{model_name}] best trial {best['trial']}: val macro F1 {best['val_macro_f1']:.4f}")
    for seed in proto["eval_seeds"]:
        if raw_path(task_name, model_name, seed).exists() and \
                (RESULTS / "seed_runs" / task_name / model_name / f"seed{seed}.json").exists():
            log(f"[{task_name}/{model_name}] seed {seed}: already done, skipped")
            continue
        trial_times = [read_json(p)["seconds"] for p in (RESULTS / "tuning" / task_name / model_name).glob("trial_*.json")]
        if trial_times and time_left() < 1.3 * max(trial_times):
            raise Deferred(f"{task_name}/{model_name}: not enough time for seed {seed}")
        final_seed(task_name, task, model_name, mcfg, data, best, seed, proto, device)
        torch.cuda.empty_cache()


def main():
    global LOG
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], required=True)
    ap.add_argument("--jobs", required=True, help="JSON list of {task, model}")
    ap.add_argument("--worker", default="worker")
    ap.add_argument("--overrides", default="{}", help="JSON protocol overrides (n_trials, eval_seeds)")
    args = ap.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    LOG = RESULTS / "logs" / f"{args.worker}.log"
    LOG.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[{args.worker}] device {device} "
        f"{torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}, torch {torch.__version__}")
    failures = 0
    for job in json.loads(args.jobs):
        with Timer() as t:
            try:
                run_job(job, args.mode, device, json.loads(args.overrides))
                status = "ok"
            except Deferred as e:
                status = "deferred"
                log(f"[{job['task']}/{job['model']}] DEFERRED: {e}")
            except Exception:  # noqa: BLE001 - record and continue with the next job
                failures += 1
                status = "error"
                err = RESULTS / "errors" / f"{job['task']}__{job['model']}.txt"
                err.parent.mkdir(parents=True, exist_ok=True)
                err.write_text(traceback.format_exc(), encoding="utf-8")
                log(f"[{job['task']}/{job['model']}] FAILED, traceback in {err}")
        timing = RESULTS / "job_times" / f"{job['task']}__{job['model']}{'__check' if job.get('check_cache') else ''}.json"
        prev = read_json(timing)["sessions"] if timing.exists() else []
        write_json({"task": job["task"], "model": job["model"], "worker": args.worker,
                    "sessions": prev + [{"mode": args.mode, "seconds": t.seconds, "status": status}]}, timing)
        torch.cuda.empty_cache()
        if status == "deferred":
            log(f"[{args.worker}] session deadline reached; remaining jobs left for the next session")
            break
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
