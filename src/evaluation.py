from pathlib import Path

import matplotlib.pyplot as plt


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

def evaluate_model(y_true, y_pred):
    metrics = {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            pos_label="anomaly",
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            pos_label="anomaly",
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            pos_label="anomaly",
            zero_division=0,
        )
    }
    return metrics

def evaluate_predictions(y_true, y_pred, model_name, section="00", save_path=None):
    results = evaluate_model(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=["normal", "anomaly"])

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Anomaly"]
    )

    disp.plot()

    plt.title(f"Section {section}: {model_name}")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    #plt.show()
    plt.close()

    return results