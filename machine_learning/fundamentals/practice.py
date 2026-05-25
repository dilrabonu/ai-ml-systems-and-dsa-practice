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

# Elastic Net L1+L2

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np

# 100 ta feature'li ma'lumot, faqat 10 tasi haqiqatan muhim
X, y, true_coef = make_regression(
    n_samples=200, n_features=100, n_informative=10,
    noise=10, coef=True, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

modellar = {
    "Regularizationsiz": LinearRegression(),
    "Ridge (L2, λ=1)": Ridge(alpha=1.0),
    "Ridge (L2, λ=10)": Ridge(alpha=10.0),
    "Lasso (L1, λ=1)": Lasso(alpha=1.0),
}

for nom, model in modellar.items():
    model.fit(X_train, y_train)
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    nolga_teng = np.sum(np.abs(model.coef_) < 0.01)
    
    print(f"{nom}")
    print(f"  Train R²: {train_r2:.3f} | Test R²: {test_r2:.3f}")
    print(f"  Nolga teng koeffitsientlar: {nolga_teng}/100")
    print()

# Dropout
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.dropout = nn.Dropout(p=0.5)    # 50% neyronlarni o'chir
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)                  # faqat train paytida ishlaydi
        return self.fc2(x)