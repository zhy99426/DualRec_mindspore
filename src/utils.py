import mindspore as ms
from mindspore import ops

def mrr(scores, labels):
    sort = ops.Sort(axis=1)
    rank = sort(-scores)[1]
    hits = ops.gather_elements(labels, 1, rank)
    idx = ops.cast(ops.nonzero(hits == 1)[:,1], ms.float32)
    mrr = 1 / (idx + 1)
    return mrr.mean()


def recalls_and_ndcgs_for_ks(scores, labels, ks):
    metrics = {}
    answer_count = labels.sum(1)
    answer_count_float = ops.cast(answer_count, ms.float32)
    labels_float = ops.cast(labels, ms.float32)
    sort = ops.Sort(axis=1)
    rank = sort(-scores)[1]
    cut = rank
    for k in sorted(ks, reverse=True):
        cut = cut[:, :k]
        hits = ops.gather_elements(labels_float, 1, cut)
        metrics["Recall@%d" % k] = (hits.sum(1) / answer_count_float.expand_dims(-1)).mean()

        position = ms.numpy.arange(2, 2 + k, dtype=ms.float32)
        weights = 1 / ms.numpy.log2(position)
        dcg = (hits * weights).sum(1)
        idcg = ops.stack([weights[: min(n, k)].sum() for n in answer_count])
        ndcg = (dcg / idcg.expand_dims(-1)).mean()
        metrics["NDCG@%d" % k] = ndcg

    return metrics


def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]
