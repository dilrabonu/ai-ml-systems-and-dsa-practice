from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Disbalans ma'lumot: 95% sog'lom, 5% kasal
X, y = make_classification(
    n_samples=1000, n_features=20, weights=[0.95, 0.05],
    random_state=42
)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y)

model = LogisticRegression(max_iter=1000)
model.fit(X_tr, y_tr)
y_pred = model.predict(X_te)
y_proba = model.predict_proba(X_te)[:, 1]   # ehtimollik

print("Confusion Matrix:")
print(confusion_matrix(y_te, y_pred))
print()
print(f"Accuracy:  {accuracy_score(y_te, y_pred):.3f}")
print(f"Precision: {precision_score(y_te, y_pred):.3f}")
print(f"Recall:    {recall_score(y_te, y_pred):.3f}")
print(f"F1-score:  {f1_score(y_te, y_pred):.3f}")
print(f"ROC-AUC:   {roc_auc_score(y_te, y_proba):.3f}")
print()
print(classification_report(y_te, y_pred))

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_true = np.array([100, 200, 300, 400, 500])     # haqiqiy uy narxlari (ming$)
y_pred = np.array([110, 190, 320, 380, 510])

print(f"MAE:  {mean_absolute_error(y_true, y_pred):.1f} ming$")
print(f"MSE:  {mean_squared_error(y_true, y_pred):.1f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.1f} ming$")
print(f"R²:   {r2_score(y_true, y_pred):.3f}")