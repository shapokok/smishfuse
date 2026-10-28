"""Canonical model order and display names for tables and reports (steps 2 and 3)."""

MODEL_ORDER = ["logreg", "linear_svm", "random_forest", "xgboost",
               "textcnn", "bilstm", "distilbert", "bert", "qwen_fewshot"]

MODEL_NAMES = {
    "logreg": "TF-IDF + LogReg",
    "linear_svm": "TF-IDF + LinearSVM",
    "random_forest": "TF-IDF + RandomForest",
    "xgboost": "TF-IDF + XGBoost",
    "textcnn": "TextCNN",
    "bilstm": "BiLSTM",
    "distilbert": "DistilBERT",
    "bert": "BERT",
    "qwen_fewshot": "Qwen2.5-7B 8-shot",
}

SHORT_NAMES = {
    "logreg": "LogReg", "linear_svm": "LinearSVM", "random_forest": "RandomForest", "xgboost": "XGBoost",
    "textcnn": "TextCNN", "bilstm": "BiLSTM", "distilbert": "DistilBERT", "bert": "BERT",
    "qwen_fewshot": "Qwen2.5-7B 8-shot",
}

STEP2_MODELS = MODEL_ORDER[:4]
STEP3_MODELS = MODEL_ORDER[4:]


def ordered(models) -> list[str]:
    models = set(models)
    return [m for m in MODEL_ORDER if m in models]
