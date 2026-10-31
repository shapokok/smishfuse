# Paper plan (step 7): draft for Applied Mathematics & Information Sciences (AMIS)

Status: plan only, for the author's review. No new experiments; every number will be taken from `results/` and
`reports/` and checked by `paper/check_numbers.py` (to be written with the first section).

## 0. Template and journal rules (found)

- Natural Sciences Publishing, AMIS author page: https://www.naturalspublishing.com/show.asp?JorID=1&pgid=3
- LaTeX sample: http://www.naturalspublishing.com/files/pages/45c843tvg4gg8h.zip (`NSP1.cls`, `paper.tex` of 2024);
  preparation guide (PDF): http://www.naturalspublishing.com/files/pages/wai25b68b7q2t2.pdf
- Rules that shape the draft:
  - LaTeX, at most 25 pages (journal layout);
  - title of at most two lines, without numbers or abbreviations (so no "SMS" in the title);
  - abstract: the page gives both "300-400 words" and "no more than 200 words"; I will keep <= 200 words,
    without references or abbreviations;
  - 4-5 keywords in alphabetical order;
  - prescribed sections 1. Introduction, 2. Materials and Methods, 3. Results and Discussion,
    Acknowledgments, References, then declarations (Funding, Competing interests, Availability of data and
    material, Code availability, Author contributions);
  - references: numbered in the text [1]; entries in the style of the sample
    ("A. Author, B. Author, Journal Abbrev. **vol**, pages (year)."); the page also says "alphabetical",
    which contradicts numbered citations: I will use citation order;
  - figures: vector (EPS with embedded fonts), 8 or 16 cm wide, lettering 8-12 pt; captions start with
    bold "Fig." without a final full stop;
  - tables numbered in citation order, footnotes with superscript letters, asterisks for significance.
- LaTeX is not installed locally (no `pdflatex`), so the draft cannot be compiled here yet.

## 1. Working title and abstract plan

Title (two lines, no abbreviations): "A Calibrated Character-Level Cascade for Robust and Fast Detection of
Fraudulent Short Text Messages"

Abstract (<= 200 words): problem; leakage/duplicates and obfuscation gaps; method (calibrated character
n-gram linear model + threshold rule deferring to a transformer); frozen protocol, 5 seeds, three corpora;
main results stated with their significance status; latency; Kazakhstan case study as a qualitative
demonstration.

Keywords (alphabetical): cascade classification; character n-grams; obfuscation robustness; phishing
detection; statistical significance.

## 2. Structure mapped to the journal sections

The user's 10-part structure is kept as subsections inside the journal's prescribed sections.

**1. Introduction** (problem; gaps: leakage and duplicates in public corpora, robustness to character
obfuscation, inference cost; 4 contributions):
1. a frozen evaluation protocol on deduplicated corpora with re-implemented published methods in their own
   settings, paired bootstrap / McNemar with Holm correction;
2. a two-stage cascade (calibrated character linear model, transformer on low-confidence messages) with the
   threshold chosen by an explicit validation criterion;
3. robustness, low-data, transfer and cost analysis, including character obfuscation types never seen in
   training;
4. a qualitative Kazakhstan case study (Russian / Kazakh / English, translation, adaptation).

**1.1 Related work** (smishing and SMS spam; transformers versus character models; robustness to
obfuscation and adversarial misspellings; cascades and early exit; dataset duplication and leakage).

**2. Materials and Methods**
- 2.1 Problem statement and notation (messages, labels Y = {ham, smishing, spam}, probabilistic classifiers,
  macro F1).
- 2.2 Data: Mendeley SMS Phishing, MOZ-Smishing, Super SMS (clean variant); deduplication (exact + MinHash
  near-duplicates, Jaccard >= 0.9 over character 5-grams), cross-corpus overlap, majority-vote label
  resolution, stratified 70/10/20 split; the 40 smishing/spam conflicting pairs.
- 2.3 Obfuscation model: a probabilistic string transformation (formal definition below), TRAIN and
  HELD-OUT operation families, training augmentation.
- 2.4 Lite model: character 2-5-gram TF-IDF, linear SVM, sigmoid (Platt) calibration.
- 2.5 Full model: the cascade decision rule with threshold tau; choice of tau as a validation optimisation.
- 2.6 Cost of errors L_r.
- 2.7 Statistical testing: paired bootstrap of the macro F1 difference, McNemar test, Holm step-down.
- 2.8 Experimental protocol: baselines of steps 2-3 (TF-IDF + LogReg / LinearSVM / RandomForest / XGBoost,
  TextCNN, BiLSTM, DistilBERT, BERT, Qwen2.5-7B 8-shot), Optuna tuning on val (20 trials), 5 seeds,
  freezing of the final configuration before reading test (commits f209949, 087615d), re-implementation of
  Mahmud et al. 2024 and Shim et al. 2024 in their settings, hardware and timing protocol.

**3. Results and Discussion**
- 3.1 Main comparison on mendeley_3class (clean test) and super_sms_clean_binary.
- 3.2 Settings of the published methods with significance (incl. the post-hoc comparison of the character
  SVM, decision 20, marked post-hoc).
- 3.3 Robustness to obfuscation (TRAIN and HELD-OUT families).
- 3.4 Low data and transfer to MOZ (Portuguese).
- 3.5 Cost of errors.
- 3.6 Speed and the quality-latency Pareto front (real end-to-end measurement of the cascade).
- 3.7 Analysis: cascade ablations (SVM only, DistilBERT only, no augmentation, averaging); SmishFuse v1 / v2
  as a negative result (validation only); spam error analysis; inflation from duplicates
  (and its confounding with conflicting labels).
- 3.8 Case study: Kazakhstan (42 author-written cases, translation with NLLB, adaptation with a synthetic
  LLM-assisted set, 4 public real messages), explicitly a qualitative demonstration; post-hoc parts marked.
- 3.9 Limitations.

**4. Conclusion**

Acknowledgments (placeholder); declarations (Funding, Competing interests, Availability of data and material,
Code availability, Author contributions; placeholders where author input is needed); References.

Appendix (if the 25-page limit allows; otherwise supplementary material): full significance tables,
per-task tables of steps 2-3, SmishFuse v1 / v2 validation tables, hyperparameters, case-study per-message
tables, PR curves, confusion matrix.

## 3. Formal part (2.1-2.7)

- Classifier f: X -> Δ^{K-1}, prediction ŷ(x) = argmax_k f_k(x); macro F1 = (1/K) Σ_k 2P_kR_k/(P_k+R_k).
- Obfuscation O_{λ,S}: for a string x with n non-space characters and an operation set S
  (TRAIN: substitution within look-alike groups, space insertion; HELD-OUT: Cyrillic homoglyphs, letter
  doubling, vowel deletion), let E(x,S) be the positions where at least one operation applies; draw
  m = min(round(λ n), |E|) positions uniformly without replacement, and at each apply an operation drawn
  uniformly from those applicable; the random stream is seeded by (seed, λ, hash(message id)), which makes
  the map deterministic per message. Augmentation: with probability p = 0.5 apply O_{λ,S_TRAIN} with
  λ ~ U(0, 0.2), a new draw every epoch.
- Lite: g(x) = σ(A w^T φ(x) + B) per class (Platt scaling fitted on out-of-fold decisions, 5 folds), φ =
  TF-IDF of character 2-5-grams.
- Cascade: ŷ_τ(x) = argmax f^(1)(x) if max_k f^(1)_k(x) >= τ, else argmax f^(2)(x); routed share
  ρ(τ) = (1/n) Σ 1[max_k f^(1)_k(x_i) < τ].
- Threshold: τ* = min argmax_{τ ∈ T} (1/|S|) Σ_{s ∈ S} F1_macro(ŷ^{(s)}_τ; D_val), T = {0} ∪ {max_k f^(1)_k(x):
  x ∈ D_val} ∪ {1 + ε}; ties to the smallest τ (fewest stage-2 calls).
- Cost: L_r = (r·FN + FP)/n on the ham-versus-malicious view, r ∈ {1, 5, 10}.
- Paired bootstrap: B = 10 000 multinomial resamples of the test set, Δ* = F1(a) - F1(b) on each, 95 % percentile
  interval, two-sided p-value as implemented in `src/eval/stats.py` (formula copied exactly from the code).
- McNemar: b, c discordant counts; exact binomial test if b + c < 25, otherwise χ² = (|b - c| - 1)²/(b + c), 1 d.f.
- Holm: sorted p_(1) <= ... <= p_(m), adjusted p_(i) = max_{j<=i} min(1, (m - j + 1) p_(j)), applied within each
  condition, separately for each test.

## 4. Tables (main text: 8) and their sources

| # | Content | Source |
|---|---|---|
| T1 | Corpora after deduplication: messages per split and class, removed duplicates and conflicts | `reports/step1_data_report.md`, `reports/stats/` |
| T2 | Main comparison, mendeley_3class clean test: all models, macro F1, accuracy, F1 smishing / spam, PR-AUC, ECE (5 seeds); super_sms_clean_binary macro F1 as a second column block | `results/final/metrics.csv` |
| T3 | Settings of the published methods (Mahmud: ham vs malicious; Shim: smishing vs spam), clean and HELD-OUT 5/10/20 %, claimed numbers (from the papers), significance of the cascade and (post-hoc) of the character SVM | `results/final/metrics.csv`, `significance.csv`, `posthoc_significance.csv`, `results/step5a/published_claims.yaml` |
| T4 | Robustness: macro F1 under TRAIN and HELD-OUT obfuscation, 5/10/20 %, selected models | `results/final/metrics.csv` |
| T5 | Low data (1/5/10/25 % of train) and transfer to MOZ (zero-shot, 10 % fine-tuning) | `results/final/metrics.csv` |
| T6 | Efficiency and cost: p50 / p95 latency, throughput, size; L_r for r = 1, 5, 10 | `results/final/latency.csv`, `cascade_end_to_end.json`, `metrics.csv` |
| T7 | Cascade ablations and routed share; significance of the cascade against the best baseline per condition | `results/final/metrics.csv`, `significance.csv`, `tau.json` |
| T8 | Duplicate inflation: dirty versus clean 5-fold CV (accuracy, macro F1) for LinearSVM, LogReg and the re-implemented methods | `results/step5a/inflation/` |

## 5. Figures (main text: 6) and their sources

| # | Content | Source |
|---|---|---|
| F1 | Cascade diagram (decision rule with τ, routed share) | new TikZ drawing, no data |
| F2 | Robustness to obfuscation (TRAIN / HELD-OUT lines) | `reports/figures/final/obfuscation_robustness.pdf` |
| F3 | Learning curve (low data) | `reports/figures/final/learning_curve.pdf` |
| F4 | Quality versus latency, Pareto front | `reports/figures/final/quality_vs_latency.pdf` |
| F5 | Share routed to the transformer versus macro F1 | `reports/figures/final/cascade_share_vs_f1.pdf` |
| F6 | Kazakhstan adaptation curve | `reports/figures/step6b/adaptation_curve.pdf` |

Appendix figures: PR curves, confusion matrix of the cascade. All figures will be exported to EPS
(`pdftops -eps`, fonts embedded).

## 6. References

30-45 entries, each verified by DOI or arXiv ID before inclusion (`paper/references.bib`, converted to the
journal's `\bibitem` style). Planned groups: corpora (UCI SMS Spam Collection, Mendeley SMS Phishing,
MOZ-Smishing, Super SMS); smishing / SMS spam detection (Mishra & Soni, Sonowal, Mahmud et al., Shim et al.,
Smishing-4C, Smishtank, Roy et al., Ghourabi & Alohaly); models (SVM, linear text classification, Platt
calibration, Random Forest, XGBoost, TextCNN, LSTM, BERT, DistilBERT, RoBERTa, Qwen2.5, NLLB); robustness
(DeepWordBug, TextBugger, Pruthi et al., Bad Characters, Belinkov & Bisk); cascades and early exit (Viola &
Jones, BranchyNet, DeeBERT / Schwartz et al.); evaluation (McNemar, Holm, Efron & Tibshirani, Dietterich,
calibration by Guo et al.); duplication and leakage (Kapoor & Narayanan, Lee et al. on deduplication,
Allamanis). Anything that cannot be verified is dropped.

## 7. Number checking

`paper/check_numbers.py`: every number in the `.tex` files is extracted and must match a value computed from
the source files listed for its table / paragraph (a mapping file `paper/number_sources.yaml`); the script
fails on any unmatched number. Rounding rules: as printed in the reports (4 decimals for F1, 2 for pp, etc.).

## 8. Points for the author's decision

1. Section structure: journal-prescribed top level (Introduction / Materials and Methods / Results and
   Discussion) with the requested 10 parts as subsections (proposed above), or the 10 parts as top-level
   sections?
2. Title wording (no abbreviations allowed).
3. Abstract length: 200 words (stricter of the two rules on the page) - confirm.
4. LaTeX: I can write the source with `NSP1.cls` now; compilation needs a TeX installation (MacTeX / BasicTeX)
   locally or a remote build (e.g. Overleaf, done by the author).
