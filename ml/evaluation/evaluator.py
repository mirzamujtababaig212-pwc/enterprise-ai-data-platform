from __future__ import annotations

from dataclasses import dataclass

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class EvaluationResult:
    """
    Classification evaluation metrics.
    """

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None

    def as_dict(self) -> dict[str, float]:
        metrics = {
            "validation_accuracy": self.accuracy,
            "validation_precision": self.precision,
            "validation_recall": self.recall,
            "validation_f1": self.f1,
        }

        if self.roc_auc is not None:
            metrics["validation_roc_auc"] = self.roc_auc

        return metrics


class ModelEvaluator:
    """
    Evaluates binary classification models.
    """

    @staticmethod
    def evaluate(
        model,
        X_test,
        y_test,
    ) -> EvaluationResult:

        predictions = model.predict(X_test)

        accuracy = float(
            accuracy_score(
                y_test,
                predictions,
            )
        )

        precision = float(
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            )
        )

        recall = float(
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            )
        )

        f1 = float(
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            )
        )

        roc_auc: float | None = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_test)

            if probabilities.shape[1] == 2:
                try:
                    roc_auc = float(
                        roc_auc_score(
                            y_test,
                            probabilities[:, 1],
                        )
                    )
                except ValueError:
                    roc_auc = None

        return EvaluationResult(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            roc_auc=roc_auc,
        )
