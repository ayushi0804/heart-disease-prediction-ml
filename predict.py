"""Standalone heart-disease risk predictor.
Usage:  python predict.py          (needs heart_model.joblib in the same folder)
Educational project only - NOT a medical device.
"""
import sys
import joblib
import pandas as pd

LIMITS = {
    "age": (1, 120), "sex": (0, 1), "cp": (0, 3), "trestbps": (60, 260), "chol": (80, 700),
    "fbs": (0, 1), "restecg": (0, 2), "thalach": (50, 250), "exang": (0, 1),
    "oldpeak": (0, 10), "slope": (0, 2), "ca": (0, 3), "thal": (1, 3),
}
PROMPTS = [
    ("age", int, "Age in years"),
    ("sex", int, "Sex (0 = female, 1 = male)"),
    ("cp", int, "Chest pain type (0-3)"),
    ("trestbps", int, "Resting blood pressure (mm Hg)"),
    ("chol", int, "Serum cholesterol (mg/dl)"),
    ("fbs", int, "Fasting blood sugar > 120 mg/dl? (1 = yes, 0 = no)"),
    ("restecg", int, "Resting ECG result (0, 1, 2)"),
    ("thalach", int, "Maximum heart rate achieved"),
    ("exang", int, "Exercise-induced angina (1 = yes, 0 = no)"),
    ("oldpeak", float, "ST depression induced by exercise (e.g. 1.4)"),
    ("slope", int, "Slope of peak exercise ST segment (0, 1, 2)"),
    ("ca", int, "Major vessels coloured by fluoroscopy (0-3)"),
    ("thal", int, "Thalassemia (1 = fixed defect, 2 = normal, 3 = reversible defect)"),
]

def ask(key, cast, text):
    lo, hi = LIMITS[key]
    while True:
        try:
            v = cast(input(f"{text} [{lo}-{hi}]: ").strip())
        except ValueError:
            print("  Please enter a number."); continue
        if lo <= v <= hi:
            return v
        print(f"  Value must be between {lo} and {hi}.")

def main():
    bundle = joblib.load("heart_model.joblib")
    model, feats, thr = bundle["model"], bundle["features"], bundle["threshold"]
    vals = {k: ask(k, c, t) for k, c, t in PROMPTS}
    p = float(model.predict_proba(pd.DataFrame([vals])[feats])[0, 1])
    print(f"\nEstimated probability of heart disease: {p:.1%}")
    print("Result:", "HIGH risk - please consult a doctor" if p >= thr else "LOW risk - heart disease not indicated")
    print("(Educational tool only, not a medical diagnosis.)")

if __name__ == "__main__":
    sys.exit(main())
