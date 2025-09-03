from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# ===============================
# 1) Dataset (texts + labels)
# ===============================
texts = ["I love this", "This is awful", "This is a good movie", "This is a bad movie", "This is a good book", "This is not a good book"]
labels = ["positive", "negative", "positive", "negative", "positive", "negative"]

# ===============================
# 2) Encode labels
# ===============================
le = LabelEncoder()
y = le.fit_transform(labels)   # ["positive", "negative"] -> [1, 0]
num_labels = len(le.classes_)

# ===============================
# 3) Tokenizer + Model
# ===============================
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

id2label = {i: lab for i, lab in enumerate(le.classes_)}
label2id = {lab: i for i, lab in id2label.items()}

model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

# ===============================
# 4) Tokenize training inputs
# ===============================
encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

# ===============================
# 5) Training loop (toy example)
# ===============================
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

labels_tensor = torch.tensor(y)

for epoch in range(200):  # tiny loop; real training = thousands of steps
    optimizer.zero_grad()
    outputs = model(**encodings, labels=labels_tensor)
    loss = outputs.loss
    loss.backward()
    optimizer.step()

print("Training done!")

# ===============================
# 6) Inference (prediction)
# ===============================
model.eval()

# Original texts
with torch.no_grad():
    outputs = model(**encodings)
    preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
pred_labels = le.inverse_transform(preds)
print("Predicted labels for training samples:", pred_labels)

# ==== Extra Inferences ====
new_texts = [
    "The movie was fantastic!",
    "I hated the food, it was disgusting.",
    "Absolutely wonderful service and staff.",
    "The movie was good but the charachter were mediocre",
    "The book was good but the charachter were mediocre"
]

# Tokenize new inputs
new_enc = tokenizer(new_texts, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    new_outputs = model(**new_enc)
    new_preds = torch.argmax(new_outputs.logits, dim=-1).cpu().numpy()

new_pred_labels = le.inverse_transform(new_preds)
for txt, label in zip(new_texts, new_pred_labels):
    print(f"Text: {txt} --> Prediction: {label}")
