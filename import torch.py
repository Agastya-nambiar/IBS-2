import torch
import pandas as pd
import numpy as np
import os
from transformers import BertModel, BertTokenizer
from tqdm import tqdm

# ------------------------
# SETTINGS
# ------------------------
CSV_PATH = r"C:\Users\sahil\Downloads\bio\final.csv"
BATCH_SIZE = 8
MAX_LEN = 512

# ------------------------
# AUTO OUTPUT PATH (same folder as input)
# ------------------------
base_dir = os.path.dirname(CSV_PATH)
OUTPUT_CSV = os.path.join(base_dir, "protbert_features_with_seq.csv")
OUTPUT_NPY = os.path.join(base_dir, "protbert_features.npy")

# ------------------------
# LOAD DATA
# ------------------------
df = pd.read_csv(CSV_PATH)

if "seq" not in df.columns:
    raise ValueError("Column 'seq' not found in CSV")

sequences = df["seq"].astype(str).str.upper().tolist()
sequences_spaced = [" ".join(list(seq)) for seq in sequences]

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

for i in tqdm(range(0, len(sequences_spaced), BATCH_SIZE)):
    batch_sequences = sequences_spaced[i:i + BATCH_SIZE]

    inputs = tokenizer(
        batch_sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN
    )

    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state.mean(dim=1)
    all_embeddings.append(embeddings.cpu().numpy())

all_embeddings = np.vstack(all_embeddings)

print("Feature extraction complete.")
print("Feature shape:", all_embeddings.shape)

# ------------------------
# SAVE FEATURES
# ------------------------
np.save(OUTPUT_NPY, all_embeddings)

feature_columns = [f"f{i+1}" for i in range(all_embeddings.shape[1])]
features_df = pd.DataFrame(all_embeddings, columns=feature_columns)
features_df.insert(0, "sequence", sequences)

features_df.to_csv(OUTPUT_CSV, index=False)

print("\nSaved features to:", OUTPUT_CSV)
print("Saved numpy file to:", OUTPUT_NPY)