from pathlib import Path

import joblib

from src.utils.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_artifacts() -> dict:
    """
    Load all fitted inference artifacts required for prediction.

    The artifacts are loaded from paths defined in config.yaml.
    No training or fitting is performed.
    """
    config = load_config()

    feature_list_path = PROJECT_ROOT / config["paths"]["feature_list"]
    preprocessor_path = PROJECT_ROOT / config["paths"]["preprocessor"]
    model_path = PROJECT_ROOT / config["paths"]["model"]
    threshold_path = PROJECT_ROOT / config["paths"]["threshold"]

    required_files = {
        "feature_list": feature_list_path,
        "preprocessor": preprocessor_path,
        "model": model_path,
        "threshold": threshold_path,
    }

    missing_files = [
        f"{name}: {path}"
        for name, path in required_files.items()
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Required inference artifacts are missing:\n"
            + "\n".join(missing_files)
        )

    with open(feature_list_path, "r", encoding="utf-8") as file:
        feature_list = [
            line.strip()
            for line in file
            if line.strip()
        ]

    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)
    threshold_artifact = joblib.load(threshold_path)

    threshold = float(threshold_artifact["threshold"])

    return {
        "feature_list": feature_list,
        "preprocessor": preprocessor,
        "model": model,
        "threshold": threshold,
    }
"""
                 config.yaml
                     │
                     ▼
              ┌─────────────┐
              │ load_config  │
              └──────┬──────┘
                     │
                     ▼
              معرفة أماكن الملفات
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
 feature_list   preprocessor    model
        │            │            │
        └────────────┼────────────┘
                     │
                     ▼
                 threshold
                     │
                     ▼
              load_artifacts()
                     │
                     ▼
             جميع الـ artifacts
                     │
                     ▼
                  predict
"""