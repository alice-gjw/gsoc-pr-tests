"""Training simple model to practice serving for deployment"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.pipeline import Pipeline

train_data = fetch_20newsgroups(subset="train")
test_data = fetch_20newsgroups(subset="test")

X_train = train_data.data
y_train = train_data.target
X_test = test_data.data
y_test = test_data.target

print(f"Training samples: {len(X_train)}")
print(f"Test samples:     {len(X_test)}")
print(f"Categories:       {train_data.target_names}")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
X_train_vectors = embedder.encode(X_train)
X_test_vectors = embedder.encode(X_test)

lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))                
])

lr_pipeline.fit(X_train_vectors, y_train)

model_path = "data/model/lr_pipeline.joblib"
joblib.dump({
    "embedder": embedder,
    "pipeline": lr_pipeline,
    "categories": train_data.target_names}, 
    model_path)


