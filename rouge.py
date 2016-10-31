import os
import sys
import json
from pyrouge import Rouge155 

if len(sys.argv) < 4:
    sys.exit(
        'Usage: python rouge.py <decode-file-path> <ref-file-path> <num-of-keywords>\n')

dec_file = sys.argv[1]
ref_file = sys.argv[2]
num_kw = int(sys.argv[3])

def top_k(l, k):
    return ','.join(l[7:].split(' , ')[:k]).strip()

rouge = Rouge155(n_words=20)
metrics = (
    'rouge_1_precision', 'rouge_1_recall', 'rouge_1_f_score',
    'rouge_2_precision', 'rouge_2_recall', 'rouge_2_f_score',
)
dec, ref = [], []
with open(dec_file, 'r') as f:
    dec += f.readlines()

with open(ref_file, 'r') as f:
    ref += f.readlines()

n_lines = min(len(dec), len(ref))
scores = {}
for i in xrange(n_lines):
    ref_line = top_k(ref[i], num_kw)
    dec_line = top_k(dec[i], num_kw)
    score = rouge.score_summary(dec_line, {'A': ref_line})
    for m in metrics:
        s = scores.setdefault(m, 0.)
        scores[m] = ((i - 1) * s + score[m])/(i + 1)
print json.dumps(scores, indent=4)
