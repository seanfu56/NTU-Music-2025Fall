import sys
import json


answer_path = sys.argv[1]
pred_path = sys.argv[2]

with open(answer_path, 'r') as f:
    answers = json.load(f)

with open(pred_path, 'r') as f:
    preds = json.load(f)


top1_correct = 0
top3_correct = 0

for k in list(preds.keys()):
    ans = answers[int(k)-1]
    pred = preds[k]
    if ans == pred[0]:
        top1_correct += 1
        top3_correct += 1

    elif ans in pred[:3]:
        top3_correct += 1


print(top1_correct/len(answers))
print(top3_correct/len(answers))
