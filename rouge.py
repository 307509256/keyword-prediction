import json
import os
from pyrouge import Rouge155 

def top_k(l, k):
    return ','.join(l[7:].split(', ')[:k]).strip()

decode_dir = './log/decode'
rouge = Rouge155(n_words=20)
metrics = (
    'rouge_1_precision', 'rouge_1_recall', 'rouge_1_f_score',
    'rouge_2_precision', 'rouge_2_recall', 'rouge_2_f_score',
)
dec, ref = [], []
for fn in os.listdir(decode_dir):
    with open('%s/%s'%(decode_dir, fn)) as f:
        if 'decode' in fn:
            dec += f.readlines()
        else:
            ref += f.readlines()

n_lines = min(len(dec), len(ref))
scores = {}
for i in xrange(n_lines):
    ref_line = top_k(ref[i], 5)
    dec_line = top_k(dec[i], 5)
    score = rouge.score_summary(dec_line, {'A': ref_line})
    for m in metrics:
        s = scores.setdefault(m, 0.)
        scores[m] = ((i - 1) * s + score[m])/(i + 1)
print json.dumps(scores, indent=4)
