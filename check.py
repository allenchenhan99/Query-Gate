import json
from collections import Counter, defaultdict

cat_label = defaultdict(Counter)

with open("dolly_processed.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        x = json.loads(line)
        cat_label[x["category"]][int(x["label"])] += 1

for cat in sorted(cat_label):
    print(cat, dict(cat_label[cat]))
