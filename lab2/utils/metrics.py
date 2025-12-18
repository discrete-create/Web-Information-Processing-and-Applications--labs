import torch
import numpy as np


# KG 链接预测指标

def get_ranks_from_scores(all_scores, true_indices, larger_is_better=False):

    if isinstance(all_scores, torch.Tensor):
        scores = all_scores.detach().cpu().numpy()
    else:
        scores = np.asarray(all_scores)

    true_indices = np.asarray(true_indices, dtype=np.int64)
    num_samples = scores.shape[0]

    ranks = np.zeros(num_samples, dtype=np.int64)

    for i in range(num_samples):
        s = scores[i]
        t_idx = true_indices[i]
        target_score = s[t_idx]

        if larger_is_better:
            better = np.sum(s > target_score)
        else:
            better = np.sum(s < target_score)

        ranks[i] = int(better) + 1  # rank 从 1 开始

    return ranks


def kg_metrics_from_ranks(ranks, Ks=(1, 3, 10)):

    ranks = np.asarray(ranks, dtype=np.float32)

    mr = ranks.mean()
    mrr = (1.0 / ranks).mean()

    metrics = {
        'MR': mr,
        'MRR': mrr,
    }
    for k in Ks:
        metrics[f'Hits@{k}'] = np.mean((ranks <= k).astype(np.float32))

    return metrics


def calc_kg_metrics(all_scores, true_indices, Ks=(1, 3, 10), larger_is_better=False):

    ranks = get_ranks_from_scores(all_scores, true_indices, larger_is_better=larger_is_better)
    metrics = kg_metrics_from_ranks(ranks, Ks=Ks)
    return metrics
