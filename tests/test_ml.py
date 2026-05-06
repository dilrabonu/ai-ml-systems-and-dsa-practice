import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "age":   [25, 30, np.nan, 45, 50],
        "city":  ["NY", "LA", "NY", None, "SF"],
        "score": [0.1, 0.5, 0.3, 0.9, 0.7],
    })

def make_preprocessor():
    num = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("sc",  StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh",  OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num, ["age", "score"]),
                              ("cat", cat, ["city"])])

def test_no_nans_after_transform(sample_df):
    """Imputers must eliminate all NaNs."""
    X = make_preprocessor().fit_transform(sample_df)
    assert not np.isnan(X).any()

def test_numeric_columns_zero_mean(sample_df):
    """StandardScaler must center numerics."""
    X = make_preprocessor().fit_transform(sample_df)
    assert abs(X[:, 0].mean()) < 1e-9
    assert abs(X[:, 1].mean()) < 1e-9

def test_handles_unknown_category_at_inference(sample_df):
    """The #1 sklearn production failure mode: unseen category."""
    pre = make_preprocessor().fit(sample_df)
    new = pd.DataFrame({"age": [40], "city": ["Tokyo"], "score": [0.5]})
    X = pre.transform(new)         # must not raise
    assert X.shape[0] == 1