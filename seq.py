from Bio.Align import PairwiseAligner

seq1 = "AGTACGCA"
seq2 = "TATGC"

aligner = PairwiseAligner()
aligner.mode = "global"   # Needleman-Wunsch
aligner.match_score = 1
aligner.mismatch_score = 0
aligner.open_gap_score = 0
aligner.extend_gap_score = 0

alignments = aligner.align(seq1, seq2)

for alignment in alignments:
    print(alignment)
