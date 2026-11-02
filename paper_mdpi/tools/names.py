"""English model and test-set names shared by the tables and figures of the paper."""

MODELS = {
    "cascade": "Cascade",
    "svm_char_aug": "Char SVM, aug. (stage 1)",
    "svm_char": "Char SVM",
    "distilbert_aug": "DistilBERT, aug. (stage 2)",
    "distilbert": "DistilBERT, stage-2 model, no aug.",
    "cascade_noaug": "Cascade without aug.",
    "average": "Average of the two stages",
    "smishfuse_v1": "Gated fusion v1",
    "smishfuse_v2": "Gated fusion v2, aug.",
    "smishfuse_v2_noaug": "Gated fusion v2",
    "bert": "BERT",
    "logreg": "TF-IDF + LR",
    "linear_svm": "TF-IDF + SVM",
    "svm_step2": "TF-IDF + SVM",
    "random_forest": "TF-IDF + RF",
    "xgboost": "TF-IDF + XGBoost",
    "textcnn": "TextCNN",
    "bilstm": "BiLSTM",
    "distilbert_step3": "DistilBERT",
    "qwen_fewshot": "Qwen2.5-7B, few-shot",
    "mahmud2024_cnn_bigru": "Mahmud et al., CNN-BiGRU",
    "shim2024_bert_base": "Shim et al., BERT",
    "shim2024_distilbert_base": "Shim et al., DistilBERT",
    "shim2024_roberta_base": "Shim et al., RoBERTa",
}

SETS = {"clean": "clean", "train05": "TR 5\\%", "train10": "TR 10\\%", "train20": "TR 20\\%",
        "heldout05": "HO 5\\%", "heldout10": "HO 10\\%", "heldout20": "HO 20\\%",
        "zeroshot": "zero-shot", "finetune10": "fine-tuned"}

# plain-text variants for figures (no LaTeX escapes)
SETS_PLAIN = {k: v.replace("\\%", "%") for k, v in SETS.items()}


def name(model: str) -> str:
    return MODELS.get(model, model)
