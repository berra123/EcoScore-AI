from __future__ import annotations

from pathlib import Path
from math import sqrt
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "production_data.csv"
MODEL_DIR = BASE_DIR / "models"

ENCODER_PATH = MODEL_DIR / "encoder.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

GAS_MODEL_PATH = MODEL_DIR / "xgboost_gas_model.pkl"
ELEC_MODEL_PATH = MODEL_DIR / "xgboost_elec_model.pkl"

# =============================================================================
# COLUMNS
# =============================================================================

FEATURE_COLUMNS = [
    "Urun_Tipi",
    "Siparis_Adedi",
    "Hedef_Gramaj_g",
    "Geri_Donusumlu_Kagit_Orani",
    "Hammadde_Nem_Orani",
    "Firin_Hizi_m_dk",
    "Firin_Sicakligi_C",
]

TARGET_GAS = "Dogalgaz_Tuketimi_m3"
TARGET_ELEC = "Elektrik_Tuketimi_kWh"

CATEGORICAL_FEATURES = ["Urun_Tipi"]

NUMERIC_FEATURES = [
    "Siparis_Adedi",
    "Hedef_Gramaj_g",
    "Geri_Donusumlu_Kagit_Orani",
    "Hammadde_Nem_Orani",
    "Firin_Hizi_m_dk",
    "Firin_Sicakligi_C",
]


def create_model() -> XGBRegressor:
    """
    Create and return a configured XGBoost regressor.

    Returns:
        XGBRegressor: Configured regression model.
    """
    return XGBRegressor(
        random_state=42,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
    )


def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load production dataset.

    Args:
        file_path: CSV file path.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If dataset cannot be found.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Veri dosyası bulunamadı:\n{file_path}"
        )

    df = pd.read_csv(file_path)

    print("✓ Veri başarıyla yüklendi.")
    print(f"Veri Boyutu : {df.shape[0]} satır x {df.shape[1]} sütun")

    missing = df.isnull().sum().sum()

    if missing > 0:
        print(f"Uyarı: Veri setinde {missing} adet eksik değer bulundu.")
    else:
        print("Eksik veri bulunmadı.")

    return df


def create_preprocessor() -> ColumnTransformer:
    """
    Create preprocessing transformer.

    Returns:
        Configured ColumnTransformer.
    """
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            (
                "num",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
        ]
    )


def preprocess_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Separate features and targets.

    Args:
        df: Input dataframe.

    Returns:
        Tuple of features, gas target and electricity target.
    """
    x = df[FEATURE_COLUMNS]
    y_gas = df[TARGET_GAS]
    y_elec = df[TARGET_ELEC]

    return x, y_gas, y_elec


def train_models(
    x: pd.DataFrame,
    y_gas: pd.Series,
    y_elec: pd.Series,
) -> Tuple[Pipeline, Pipeline, pd.DataFrame, pd.Series, pd.Series]:
    """
    Train gas and electricity models.

    Args:
        x: Features.
        y_gas: Gas target.
        y_elec: Electricity target.

    Returns:
        Trained pipelines and test datasets.
    """
    (
        x_train,
        x_test,
        y_gas_train,
        y_gas_test,
        y_elec_train,
        y_elec_test,
    ) = train_test_split(
        x,
        y_gas,
        y_elec,
        test_size=0.20,
        random_state=42,
        shuffle=True,
    )

    gas_pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("model", create_model()),
        ]
    )

    elec_pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("model", create_model()),
        ]
    )

    gas_pipeline.fit(x_train, y_gas_train)
    elec_pipeline.fit(x_train, y_elec_train)

    print("✓ Eğitim tamamlandı.")

    return (
        gas_pipeline,
        elec_pipeline,
        x_test,
        y_gas_test,
        y_elec_test,
    )


def calculate_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> Dict[str, float]:
    """
    Calculate regression metrics.

    Args:
        y_true: Actual target values.
        y_pred: Predicted target values.

    Returns:
        Dictionary containing R², MAE and RMSE metrics.
    """

    mse = mean_squared_error(
        y_true,
        y_pred,
    )

    rmse = sqrt(mse)

    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse,
    }


def evaluate_models(
    gas_pipeline: Pipeline,
    elec_pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_gas_test: pd.Series,
    y_elec_test: pd.Series,
) -> None:
    """
    Evaluate trained models.

    Args:
        gas_pipeline: Gas prediction pipeline.
        elec_pipeline: Electricity prediction pipeline.
        x_test: Test features.
        y_gas_test: Test gas values.
        y_elec_test: Test electricity values.
    """
    gas_predictions = gas_pipeline.predict(x_test)
    elec_predictions = elec_pipeline.predict(x_test)

    gas_metrics = calculate_metrics(
        y_gas_test,
        gas_predictions,
    )

    elec_metrics = calculate_metrics(
        y_elec_test,
        elec_predictions,
    )

    print("\n----------------------------------------")
    print("DOGALGAZ MODELİ")
    print(f"R²   : {gas_metrics['R2']:.4f}")
    print(f"MAE  : {gas_metrics['MAE']:.4f}")
    print(f"RMSE : {gas_metrics['RMSE']:.4f}")

    print("----------------------------------------")
    print("ELEKTRİK MODELİ")
    print(f"R²   : {elec_metrics['R2']:.4f}")
    print(f"MAE  : {elec_metrics['MAE']:.4f}")
    print(f"RMSE : {elec_metrics['RMSE']:.4f}")
    print("----------------------------------------")


def save_models(
    gas_pipeline: Pipeline,
    elec_pipeline: Pipeline,
) -> None:
    """
    Save trained models, encoder and scaler.

    Args:
        gas_pipeline: Gas pipeline.
        elec_pipeline: Electricity pipeline.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    gas_preprocessor = gas_pipeline.named_steps["preprocessor"]

    encoder = gas_preprocessor.named_transformers_["cat"]
    scaler = gas_preprocessor.named_transformers_["num"]

    joblib.dump(gas_pipeline, GAS_MODEL_PATH)
    print("✓ Gaz modeli kaydedildi.")

    joblib.dump(elec_pipeline, ELEC_MODEL_PATH)
    print("✓ Elektrik modeli kaydedildi.")

    joblib.dump(encoder, ENCODER_PATH)
    print("✓ Encoder kaydedildi.")

    joblib.dump(scaler, SCALER_PATH)
    print("✓ Scaler kaydedildi.")


def main() -> None:
    """
    Execute complete training workflow.
    """
    try:
        df = load_data(DATA_PATH)

        x, y_gas, y_elec = preprocess_data(df)

        (
            gas_pipeline,
            elec_pipeline,
            x_test,
            y_gas_test,
            y_elec_test,
        ) = train_models(
            x,
            y_gas,
            y_elec,
        )

        evaluate_models(
            gas_pipeline,
            elec_pipeline,
            x_test,
            y_gas_test,
            y_elec_test,
        )

        save_models(
            gas_pipeline,
            elec_pipeline,
        )

        print("✓ Eğitim başarıyla tamamlandı.")

    except Exception as error:
        print(f"HATA: {error}")


if __name__ == "__main__":
    main()