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

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1-qadam: avval test'ni AJRATIB QO'YAMIZ
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# 2-qadam: qolganidan train va validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.18, random_state=42, stratify=y_temp
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# 3-qadam: turli hyperparametr'larni VALIDATION'da sinaymiz
eng_yaxshi_score = 0
eng_yaxshi_n = None

for n_trees in [10, 50, 100, 200, 500]:
    model = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    model.fit(X_train, y_train)
    val_score = model.score(X_val, y_val)
    print(f"n_trees={n_trees}: val_acc={val_score:.3f}")
    
    if val_score > eng_yaxshi_score:
        eng_yaxshi_score = val_score
        eng_yaxshi_n = n_trees

# 4-qadam: eng yaxshi model'ni TRAIN + VAL da qayta o'qitamiz
final_model = RandomForestClassifier(n_estimators=eng_yaxshi_n, random_state=42)
final_model.fit(
    np.vstack([X_train, X_val]),
    np.concatenate([y_train, y_val])
)

# 5-qadam: TEST'da FAQAT BIR MARTA sinaymiz
test_score = final_model.score(X_test, y_test)
print(f"\nYakuniy test natija: {test_score:.3f}")

from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(
    RandomForestClassifier(n_estimators=100),
    X_temp, y_temp,    # test'ni ALOHIDA saqlaymiz!
    cv=cv,
    scoring='f1'
)

print(f"CV F1: {scores.mean():.3f} ± {scores.std():.3f}")

from sklearn.linear_model import LinearRegression
import numpy as np

# Uy maydoni vs narxi
X = np.array([[50], [70], [100], [150], [200]])    # m²
y = np.array([50, 75, 100, 160, 220])              # ming$

model = LinearRegression()
model.fit(X, y)

print(f"Tenglama: narx = {model.coef_[0]:.2f} × maydon + {model.intercept_:.2f}")
print(f"100 m² uy: {model.predict([[100]])[0]:.1f} ming$")

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X, y = data.data, data.target

model = LogisticRegression(max_iter=5000)
model.fit(X, y)

# Birinchi 5 ta feature'ning ahamiyati
for nom, koef in zip(data.feature_names[:5], model.coef_[0][:5]):
    print(f"{nom}: {koef:.3f}")

from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X, y)

plt.figure(figsize=(15, 8))
plot_tree(model, feature_names=data.feature_names, 
          class_names=['malign', 'benign'], filled=True)
plt.show()