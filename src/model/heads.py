import mindspore as ms
from mindspore import nn, ops


class LinearPredictionHead(nn.Cell):
    def __init__(self, args):
        super().__init__()
        self.vocab_size = args.num_items + 1
        hidden = args.d_model
        self.out = nn.Dense(hidden, self.vocab_size)

    def construct(self, x, candidates=None):
        x = self.out(x)  # B x V or M x V
        if candidates is not None:
            x = x.gather(1, candidates)  # B x C or M x C
        return x


class DotProductPredictionHead(nn.Cell):
    """share embedding parameters"""

    def __init__(self, d_model, num_items, token_embeddings):
        super().__init__()
        self.token_embeddings = token_embeddings
        self.vocab_size = num_items + 1

    def construct(self, x, candidates=None):
        if candidates is not None:  # x : B x H
            emb = self.token_embeddings(candidates)  # B x C x H
            logits = (x.expand_dims(1) * emb).sum(-1)  # B x C
        else:  # x : M x H
            emb = self.token_embeddings.embedding_table[: self.vocab_size]  # V x H
            logits = ops.matmul(x, emb.transpose(1, 0))  # M x V
            # logits_list = []
            # n = self.vocab_size // 1000
            # for i in range(n):
            #     logits_list.append(ops.matmul(x, emb.transpose(1, 0)))
        return logits
