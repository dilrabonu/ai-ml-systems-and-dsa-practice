import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Sun'iy ma'lumot yaratamiz: y = sin(x) + biroz shovqin
np.random.seed(42)
X = np.sort(np.random.rand(30, 1) * 6, axis=0)
y = np.sin(X).ravel() + np.random.randn(30) * 0.2

# 3 xil murakkablikdagi model sinaymiz
darajalar = [1, 4, 15]  # polynomial degree
nomlari = ["Underfitting (Bias yuqori)", 
           "Yaxshi balans", 
           "Overfitting (Variance yuqori)"]

X_test = np.linspace(0, 6, 100).reshape(-1, 1)

for daraja, nom in zip(darajalar, nomlari):
    model = make_pipeline(
        PolynomialFeatures(daraja),
        LinearRegression()
    )
    model.fit(X, y)
    
    train_score = model.score(X, y)
    print(f"{nom}: Train R² = {train_score:.3f}")

# Oddiy qoida:
# Train_error  Test_error    Muammo                Yechim
# ─────────────────────────────────────────────────────────────────
# Yuqori       Yuqori        Underfitting (Bias)   Modelni murakkab qil
# Past         Yuqori        Overfitting (Var)     Regularization, ko'p ma'lumot
# Past         Past          Ideal                 Hech narsa qilma :)
# Yuqori       Past          Bug bor               Kodni tekshir!

def diagnostika(train_err, test_err, baseline_err=0.1):
    if train_err > baseline_err and test_err > baseline_err:
        return "UNDERFITTING - modelni kuchaytir"
    elif train_err < baseline_err and test_err > baseline_err * 2:
        return "OVERFITTING - regularization qo'sh"
    elif train_err < baseline_err and test_err < baseline_err * 1.5:
        return "YAXSHI MODEL"
    else:
        return "Kodni tekshir, nimadir noto'g'ri"