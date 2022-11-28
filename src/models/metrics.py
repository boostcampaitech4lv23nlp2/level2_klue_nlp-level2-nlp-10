import numpy as np
import sklearn
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
np.seterr(invalid='ignore')

def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = [
        "no_relation",
        "org:top_members/employees",
        "org:members",
        "org:product",
        "per:title",
        "org:alternate_names",
        "per:employee_of",
        "org:place_of_headquarters",
        "per:product",
        "org:number_of_employees/members",
        "per:children",
        "per:place_of_residence",
        "per:alternate_names",
        "per:other_family",
        "per:colleagues",
        "per:origin",
        "per:siblings",
        "per:spouse",
        "org:founded",
        "org:political/religious_affiliation",
        "org:member_of",
        "per:parents",
        "org:dissolved",
        "per:schools_attended",
        "per:date_of_death",
        "per:date_of_birth",
        "per:place_of_birth",
        "per:place_of_death",
        "org:founded_by",
        "per:religion",
    ]
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return f1_score(labels, preds, average="micro", labels=label_indices) * 100.0


def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(probs.shape[-1])[labels]
    score = np.zeros((probs.shape[-1],))
    for c in range(probs.shape[-1]):
        if 1 in labels[:, c]:
            precision, recall, _ = precision_recall_curve(
                    labels[:, c], probs[:, c], pos_label=1)
            score[c] = sklearn.metrics.auc(recall, precision)
    
    precision, recall, _ = precision_recall_curve(
                labels.ravel(), probs.ravel())
    micro_score = sklearn.metrics.auc(recall, precision)
    
    np.where(score == 0, 1, score)
    return np.average(score) * 100.0, np.average(micro_score) * 100.0


def compute_metrics(probs, preds, labels):
    """validation을 위한 metrics function"""

    # calculate accuracy using sklearn's function
    f1 = klue_re_micro_f1(preds, labels)
    auprc, micro_auprc = klue_re_auprc(probs, labels)
    acc = accuracy_score(labels, preds)
    avg_auprc = (auprc + micro_auprc) / 2
    return {
        "micro f1 score": f1,
        "auprc": auprc,
        "micro_auprc": micro_auprc,
        "avg_auprc": avg_auprc,
        "accuracy": acc,
    }

