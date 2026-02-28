import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.4f}")

import pathlib

model_path = pathlib.Path(__file__).parent / "data/model.joblib"
model_path.parent.mkdir(exist_ok=True)
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
