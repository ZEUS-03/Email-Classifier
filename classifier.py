import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sentence_transformers import SentenceTransformer
import joblib

# Load JSON file
with open("emails.json", "r", encoding="utf-8") as f:
    data = json.load(f) 

df = pd.DataFrame(data)

df["email_text"] = df["subject"] + " " + df["body"] + " " + df["sender"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["email_text"], df["category"], test_size=0.2, random_state=42
)

# Load MiniLM model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Convert to embeddings
X_train_emb = embedder.encode(X_train.tolist(), batch_size=32, show_progress_bar=True)
X_test_emb = embedder.encode(X_test.tolist(), batch_size=32, show_progress_bar=True)

# Train Logistic Regression classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_emb, y_train)

y_pred = clf.predict(X_test_emb)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(clf, "email_classifier.pkl")
embedder.save("minilm_embedder")
