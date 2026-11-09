"""CodeAlpha Task 1: Iris Flower Classification"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 1. Load data (use pd.read_csv("Iris.csv") if you downloaded the CSV instead)
iris = load_iris(as_frame=True)
df = iris.frame
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
print(df.head(), "\n")
print(df.describe(), "\n")
print("Missing values:\n", df.isnull().sum(), "\n")

# 2. EDA
sns.pairplot(df.drop(columns="target"), hue="species")
plt.savefig("pairplot.png", dpi=120); plt.close()
plt.figure(figsize=(6, 5))
sns.heatmap(df[iris.feature_names].corr(), annot=True, cmap="coolwarm")
plt.title("Feature correlation"); plt.tight_layout()
plt.savefig("correlation.png", dpi=120); plt.close()

# 3. Split
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Compare models
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "SVM": make_pipeline(StandardScaler(), SVC()),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}
results = {}
for name, model in models.items():
    cv = cross_val_score(model, X_train, y_train, cv=5).mean()
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    results[name] = (cv, acc)
    print(f"{name:20s} CV acc = {cv:.3f} | Test acc = {acc:.3f}")

# 5. Evaluate best model
best = max(results, key=lambda k: results[k][0])
print(f"\nBest model (by CV): {best}")
pred = models[best].predict(X_test)
print(classification_report(y_test, pred, target_names=iris.target_names))
ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=iris.target_names).plot(cmap="Blues")
plt.title(f"Confusion Matrix - {best}"); plt.savefig("confusion_matrix.png", dpi=120); plt.close()

# 6. Predict a new flower
sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=iris.feature_names)
print("Sample prediction:", iris.target_names[models[best].predict(sample)[0]])
