import joblib
from sentence_transformers import SentenceTransformer

clf = joblib.load("email_classifier.pkl")
embedder = SentenceTransformer("minilm_embedder")

# Example new email
new_email = {
  "subject": "Here’s your Airtel payment receipt!",
  "body": "Dear Gautam Sharma Thank you for choosing Airtel. We have received a payment of Rs 1062.02 for your Bill payment | Recharging. Please find the payment receipt attached.Regards ,Team Airtel",
  "sender": "Airtel <update@airtel.com>"
}

# Combine text
text = new_email["subject"] + " " + new_email["body"] + " " + new_email["sender"]

# Generate embedding & predict
emb = embedder.encode([text])
print("Prediction:", clf.predict(emb)[0])
