import pandas as pd, joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

df = pd.read_csv("StressLevelDataset.csv")

FEATURES = ["sleep_quality", "study_load", "social_support", "peer_pressure",
            "extracurricular_activities", "future_career_concerns",
            "noise_level", "living_conditions", "bullying", "self_esteem"]

X, y = df[FEATURES], df["stress_level"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42),
}

scores = {}
for name, m in models.items():
    scores[name] = cross_val_score(m, X_tr, y_tr, cv=5, scoring="f1_macro").mean()
    print(f"{name}: CV macro-F1 = {scores[name]:.3f}")

best_name = max(scores, key=scores.get)
best = models[best_name].fit(X_tr, y_tr)
print("\nBest:", best_name)
print(classification_report(y_te, best.predict(X_te), target_names=["Low", "Medium", "High"]))
ConfusionMatrixDisplay.from_estimator(best, X_te, y_te, display_labels=["Low", "Medium", "High"])
plt.savefig("confusion_matrix.png", dpi=150)

# Explainability: which factors matter most
imp = permutation_importance(best, X_te, y_te, n_repeats=10, random_state=42)
pd.Series(imp.importances_mean, index=FEATURES).sort_values().plot(kind="barh", title="Feature importance")
plt.tight_layout(); plt.savefig("feature_importance.png", dpi=150)

ranges = {c: (int(df[c].min()), int(df[c].max())) for c in FEATURES}
joblib.dump({"model": best, "features": FEATURES, "ranges": ranges}, "model.joblib")