import pandas as pd
import numpy as np
from Bio.Align import PairwiseAligner
from tqdm import tqdm

# STEP 1: LOAD DATA
filepath = r"C:\Users\sahil\Downloads\bio\seq.csv"

df = pd.read_csv(
    filepath,
    header=None,
    engine="python",
    on_bad_lines="skip"
)

# If single column → only sequences
if df.shape[1] == 1:
    df.columns = ["sequence"]
    df["id"] = ["pep" + str(i) for i in range(len(df))]
    df = df[["id", "sequence"]]

# If 2 or more columns → take first two
else:
    df = df.iloc[:, :2]
    df.columns = ["id", "sequence"]

print("Original dataset size:", len(df))


# STEP 2: CLEAN SEQUENCES
df["sequence"] = df["sequence"].astype(str)
df["sequence"] = df["sequence"].str.strip()
df["sequence"] = df["sequence"].str.upper()
df["sequence"] = df["sequence"].str.replace(r"[^A-Z]", "", regex=True)

df = df[df["sequence"] != ""]
df = df.dropna()

print("After cleaning:", len(df))


# STEP 3: REMOVE EXACT DUPLICATES
df = df.drop_duplicates(subset="sequence")
print("After removing duplicates:", len(df))


# STEP 4: LENGTH FILTER (5–50)
df["length"] = df["sequence"].apply(len)
df = df[(df["length"] >= 5) & (df["length"] <= 50)]
print("After length filtering:", len(df))


# STEP 5: REMOVE NON-STANDARD AMINO ACIDS
valid_aa = set("ACDEFGHIKLMNPQRSTVWY")

def is_valid(seq):
    return all(residue in valid_aa for residue in seq)

df = df[df["sequence"].apply(is_valid)]
print("After removing invalid amino acids:", len(df))


# STEP 6: REMOVE HIGH SIMILARITY (>95%)

aligner = PairwiseAligner()
aligner.mode = "global"
aligner.match_score = 1
aligner.mismatch_score = 0
aligner.open_gap_score = -1
aligner.extend_gap_score = -0.5

def global_identity(s1, s2):
    score = aligner.score(s1, s2)
    return score / max(len(s1), len(s2))

non_redundant = []
sequences = df["sequence"].tolist()

for seq in tqdm(sequences):
    keep = True
    for existing in non_redundant:
        if global_identity(seq, existing) > 0.95:
            keep = False
            break
    if keep:
        non_redundant.append(seq)

df = df[df["sequence"].isin(non_redundant)]
print("After removing >95% similar sequences:", len(df))


# STEP 7: GLOBAL & LOCAL ALIGNMENT EXAMPLE
if len(df) >= 2:
    seq1 = df["sequence"].iloc[0]
    seq2 = df["sequence"].iloc[1]

    aligner.mode = "global"
    print("Global alignment score:", aligner.score(seq1, seq2))

    aligner.mode = "local"
    print("Local alignment score:", aligner.score(seq1, seq2))


# STEP 8: SIMILARITY MATRIX (Top 100)
def compute_similarity_matrix(seqs):
    n = len(seqs)
    matrix = np.zeros((n, n))
    aligner.mode = "global"

    for i in tqdm(range(n)):
        for j in range(i, n):
            score = aligner.score(seqs[i], seqs[j])
            identity = score / max(len(seqs[i]), len(seqs[j]))
            matrix[i][j] = identity
            matrix[j][i] = identity

    return matrix

subset = df["sequence"].tolist()[:100]

if len(subset) > 1:
    similarity_matrix = compute_similarity_matrix(subset)
    np.savetxt("similarity_matrix.csv", similarity_matrix, delimiter=",")
    print("Similarity matrix saved as similarity_matrix.csv")


# STEP 9: SAVE CLEANED DATASET
df = df.drop(columns=["length"])
df.to_csv("cleaned_nonredundant_peptides.csv", index=False)

print("\nPIPELINE COMPLETE")
