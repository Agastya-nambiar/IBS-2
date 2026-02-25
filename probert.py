import torch
import pandas as pd
import numpy as np
from transformers import BertModel, BertTokenizer
from tqdm import tqdm

# ------------------------
# SETTINGS
# ------------------------
CSV_PATH = r"C:\Users\sahil\Downloads\bio\final.csv"
BATCH_SIZE = 16
MAX_LEN = 512

# ------------------------
# LOAD DATA
# ------------------------
df = pd.read_csv(CSV_PATH)

if "seq" not in df.columns:
    raise ValueError("Column 'seq' not found in CSV")

sequences = df["seq"].astype(str).str.upper().tolist()

# Format sequences for ProtBERT (space separated)
sequences = [" ".join(list(seq)) for seq in sequences]

# ------------------------
# LOAD MODEL
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = BertModel.from_pretrained("Rostlab/prot_bert")

model.to(device)
model.eval()

# ------------------------
# FEATURE EXTRACTION
# ------------------------
all_embeddings = []

for i in tqdm(range(0, len(sequences), BATCH_SIZE)):
    batch = sequences[i:i+BATCH_SIZE]

    inputs = tokenizer(
        batch,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling (recommended)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    embeddings = embeddings.cpu().numpy()

    all_embeddings.append(embeddings)

# Combine all batches
all_embeddings = np.vstack(all_embeddings)

# ------------------------
# SAVE FEATURES
# ------------------------
np.save("protbert_features.npy", all_embeddings)
pd.DataFrame(all_embeddings).to_csv("protbert_features.csv", index=False)

print("\nFeature extraction complete.")
print("Feature shape:", all_embeddings.shape)