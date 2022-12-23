import numpy as np
import mindspore as ms
from mindspore import nn, ops
from mindspore.common import initializer
# 1D conv compression function
class MultiheadAttention(nn.Cell):
    def __init__(
        self,
        d_model: int = None,
        d_head: int = None,
        n_head: int = None,
        seg_len: int = None,
        layer_norm_eps: float = 1e-8,
        dropout: float = None,
        layer: int = 0,
    ):
        super().__init__()

        if d_model % n_head != 0:
            raise ValueError(
                f"The hidden size ({d_model}) is not a multiple of the number of attention "
                f"heads ({n_head}"
            )

        self.seg_len = seg_len
        self.n_head = n_head
        self.d_head = d_head
        self.scale = 1 / (self.d_head ** 0.5)
        self.layer = layer

        self.q = ms.Parameter(ms.Tensor(np.random.randn(d_model, self.n_head, self.d_head), ms.float32))
        self.k = ms.Parameter(ms.Tensor(np.random.randn(d_model, self.n_head, self.d_head), ms.float32))
        self.v = ms.Parameter(ms.Tensor(np.random.randn(d_model, self.n_head, self.d_head), ms.float32))
        self.r = ms.Parameter(ms.Tensor(np.random.randn(d_model, self.n_head, self.d_head), ms.float32))
        self.o = ms.Parameter(ms.Tensor(np.random.randn(d_model, self.n_head, self.d_head), ms.float32))

        self.r_r_bias = ms.Parameter(ms.Tensor(np.random.randn(self.n_head, self.d_head), ms.float32))
        self.r_w_bias = ms.Parameter(ms.Tensor(np.random.randn(self.n_head, self.d_head), ms.float32))

        self.layer_norm = nn.LayerNorm([d_model], epsilon=float(layer_norm_eps))
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def rel_shift_bnij(x, klen=-1):
        x_size = x.shape

        x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        x = x[:, :, 1:, :]
        x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        # Note: the tensor-slice form was faster in my testing than torch.index_select
        #       However, tracing doesn't like the nature of the slice, and if klen changes
        #       during the run then it'll fail, whereas index_select will be fine.
        x = x[:,:,:,ms.numpy.arange(klen, dtype=ms.int64)]

        return x

    def rel_attn_core(
        self,
        q_head,
        k_head_h,
        v_head_h,
        k_head_r,
        attn_mask=None,
        output_attentions=False,
    ):
        """Core relative positional attention operations."""
        # content based attention score
        einsum = ops.Einsum("ibnd,jbnd->bnij")
        ac = einsum((q_head + self.r_w_bias, k_head_h))

        # position based attention score
        bd = einsum((q_head + self.r_r_bias, k_head_r))
        bd = self.rel_shift_bnij(bd, klen=ac.shape[3])

        # merge attention scores and perform masking
        attn_score = (ac + bd) * self.scale


        if attn_mask is not None:
            einsum = ops.Einsum("ijbn->bnij")
            if attn_mask.dtype == ms.float16:
                attn_score = attn_score - 65500 * einsum((attn_mask,))
            else:
                attn_score = attn_score - 1e30 * einsum((attn_mask,))

        # attention probability
        attn_prob = ops.Softmax(axis=3)(attn_score)
        attn_prob = self.dropout(attn_prob)

        # attention output
        einsum = ops.Einsum("bnij,jbnd->ibnd")
        attn_vec = einsum((attn_prob, v_head_h))

        if output_attentions:
            einsum = ops.Einsum("bnij->ijbn")
            return attn_vec, einsum((attn_prob,))

        return attn_vec

    def post_attention(self, h, attn_vec, residual=True):
        """Post-attention processing. (back to `d_model`)"""
        # attn_out = attn_vec.reshape(attn_vec.shape[0], attn_vec.shape[1], -1)
        einsum = ops.Einsum("ibnd,hnd->ibh")
        attn_out = einsum((attn_vec, self.o))

        output = self.dropout(attn_out)
        if residual:
            output = output + h
        output = self.layer_norm(ops.cast(output, ms.float32))
        return output

    def construct(
        self,
        h,
        attn_mask,
        r,
        output_attentions=False,
    ):

        # Two-stream attention with relative positional encoding.
        # content based attention score
        # content-based key head
        einsum = ops.Einsum("ibh,hnd->ibnd")
        k_head_h = einsum((h, self.k))
        # content-based value head
        v_head_h = einsum((h, self.v))
        # position-based key head
        k_head_r = einsum((r, self.r))

        # h-stream
        # content-stream query head
        q_head = einsum((h, self.q))
        # core attention ops
        attn_vec = self.rel_attn_core(
            q_head,
            k_head_h,
            v_head_h,
            k_head_r,
            attn_mask=attn_mask,
            output_attentions=output_attentions,
        )

        if output_attentions:
            attn_vec, attn_prob = attn_vec

        # post processing
        output = self.post_attention(h, attn_vec)

        outputs = (output,)
        if output_attentions:
            outputs = outputs + (attn_vec,)
        return outputs


class PointWiseFeedForward(nn.Cell):
    def __init__(
        self,
        d_model: int = None,
        d_inner: int = None,
        layer_norm_eps: float = 1e-8,
        dropout: float = None,
        activation_type: str = None,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm([d_model], epsilon=float(layer_norm_eps))
        self.layer_1 = nn.Dense(d_model, d_inner)
        self.layer_2 = nn.Dense(d_inner, d_model)
        self.dropout = nn.Dropout(dropout)
        if activation_type == "relu":
            self.activation_function = nn.ReLU()
        elif activation_type == "gelu":
            self.activation_function = nn.GELU()

    def construct(self, inp):
        output = ops.cast(inp, ms.float32)
        output = self.layer_1(output)
        output = self.activation_function(output)
        output = self.dropout(output)
        output = self.layer_2(output)
        output = self.dropout(output)
        output = self.layer_norm(output + inp)
        return output


class DualRecModel(nn.Cell):
    def __init__(
        self,
        d_model: int = None,
        d_head: int = None,
        n_head: int = None,
        d_inner: int = None,
        layer_norm_eps: float = 1e-8,
        dropout: float = 0.0,
        activation_type: str = None,
        clamp_len: int = None,
        n_layer: int = None,
        num_items: int = None,
        seg_len: int = None,
        device: str = None,
        initializer_range: float = 0.02,
        reverse=False,
        return_representation=False,
        multi_scale=False,
    ):
        super().__init__()

        self.initializer_range = initializer_range

        self.d_model = d_model
        self.d_head = d_head
        self.n_head = n_head
        self.d_inner = d_inner
        self.layer_norm_eps = layer_norm_eps
        self.num_items = num_items
        self.activation_type = activation_type
        self.seg_len = seg_len
        self.clamp_len = clamp_len
        self.n_layer = n_layer
        self.return_representation = return_representation
        self.reverse = reverse

        if reverse:
            self.dir_mask = ms.Parameter(
                ms.numpy.tril(ops.ones((seg_len, seg_len), ms.int64), -1), requires_grad=False
            )
        else:
            self.dir_mask = ms.Parameter(
                ms.numpy.triu(ops.ones((seg_len, seg_len), ms.int64), 1), requires_grad=False
            )
        self.multi_scale = multi_scale
        self.omega = [2, 3, 4, 5, 7, 11, 21, 50]
        self.ms_mask = ms.Parameter(
            ops.transpose(
                ms.Tensor(
                    [
                        np.tri(seg_len, seg_len, omega - 1)
                        != np.tri(seg_len, seg_len, -omega)
                        if type(omega) == int
                        else np.tri(seg_len, seg_len, int(seg_len * omega) - 1)
                        != np.tri(seg_len, seg_len, -int(seg_len * omega))
                        for omega in self.omega
                    ], dtype=ms.float32
                ),
                (1, 2, 0)),
            requires_grad=False,
        )
        self.item_embedding = nn.Embedding(num_items + 1, d_model, padding_idx=0)


        self.attn_layers = nn.CellList(
            [
                MultiheadAttention(
                    d_model=d_model,
                    d_head=d_head,
                    n_head=n_head,
                    seg_len=seg_len,
                    layer_norm_eps=layer_norm_eps,
                    dropout=dropout,
                    layer=i,
                )
                for i in range(n_layer)
            ]
        )

        self.ff_layers = nn.CellList(
            [
                PointWiseFeedForward(
                    d_model=d_model,
                    d_inner=d_inner,
                    layer_norm_eps=layer_norm_eps,
                    dropout=dropout,
                    activation_type=activation_type,
                )
                for _ in range(n_layer)
            ]
        )
        self.dropout = nn.Dropout(dropout)

        self.device = device
        for m in self.cells_and_names():
            self._init_weights(m[1])

    def _init_weights(self, module):     
        """Initialize the weights."""
        if isinstance(module, nn.Dense):
            module.weight.set_data(initializer.initializer(initializer.Normal(sigma=self.initializer_range, mean=0.0),
                                                             module.weight.shape,
                                                             module.weight.dtype))

            # module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.set_data(initializer.initializer(initializer.Zero(),
                                                             module.bias.shape,
                                                             module.bias.dtype))
                # module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.embedding_table.set_data(initializer.initializer(initializer.Normal(sigma=self.initializer_range, mean=0.0),
                                                             module.embedding_table.shape,
                                                             module.embedding_table.dtype))
            # module.embedding_table.data.normal_(mean=0.0, std=self.initializer_range)
            # if module.padding_idx is not None:
            #     module.embedding_table[module.padding_idx].set_data(initializer.initializer(initializer.Zero()))
                # module.embedding_table.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.gamma.set_data(initializer.initializer(initializer.One(),
                                                             module.gamma.shape,
                                                             module.gamma.dtype))
            module.beta.set_data(initializer.initializer(initializer.Zero(),
                                                             module.beta.shape,
                                                             module.beta.dtype))
            # module.bias.data.zero_()
            # module.weight.data.fill_(1.0)
        elif isinstance(module, MultiheadAttention):
            for param in [
                module.q,
                module.k,
                module.v,
                module.r,
                module.o,
                module.r_r_bias,
                module.r_w_bias,
            ]:
                # param.data.normal_(mean=0.0, std=self.initializer_range)
                param.set_data(initializer.initializer(initializer.Normal(sigma=self.initializer_range, mean=0.0),
                                                             param.shape,
                                                             param.dtype))


    @staticmethod
    def positional_embedding(pos_seq, inv_freq, bsz=None):
        einsum = ops.Einsum("i,d->id")
        sinusoid_inp = einsum((pos_seq, inv_freq))
        pos_emb = ops.Concat(axis=-1)([ops.sin(sinusoid_inp), ops.cos(sinusoid_inp)])
        pos_emb = pos_emb[:, None, :]

        if bsz is not None:
            pos_emb = ops.BroadcastTo((-1, bsz, -1))(pos_emb)

        return pos_emb

    def relative_positional_encoding(self, qlen, klen, bsz=None):
        # create relative positional encoding.
        freq_seq = ms.numpy.arange(0.0, float(self.d_model), 2.0)
        inv_freq = 1 / ops.pow(10000, (freq_seq / self.d_model))

        beg, end = klen, -qlen

        fwd_pos_seq = ms.numpy.arange(float(beg), float(end), -1.0)
        if self.clamp_len > 0:
            fwd_pos_seq = fwd_pos_seq.clamp(-self.clamp_len, self.clamp_len)
        pos_emb = self.positional_embedding(fwd_pos_seq, inv_freq, bsz)

        pos_emb = pos_emb
        return pos_emb

    def construct(self, input_ids=None, input_mask=None, output_attentions=False):
        # the original code for DualRec uses shapes [len, bsz] with the batch dimension at the end
        # but we want a unified interface in the library with the batch size on the first dimension
        # so we move here the first dimension (batch) to the end

        input_ids = input_ids.transpose(1, 0)
        input_mask = input_mask.transpose(1, 0)

        qlen, bsz = input_ids.shape[0], input_ids.shape[1]
        klen = qlen
        # dealing with mask: input_mask

        data_mask = (
            input_mask[None] + input_mask[:, None] + self.dir_mask.expand_dims(-1)
        )
        if self.multi_scale and self.training:
            attn_mask = data_mask[:, :, :, None] + (1-self.ms_mask.expand_dims(-2))
        else:
            attn_mask = data_mask[:, :, :, None]
        attn_mask = ops.cast(attn_mask > 0, ms.float32)


        # Word embeddings and prepare h & g hidden states
        item_emb_k = self.item_embedding(input_ids)
        output = self.dropout(item_emb_k)

        # Positional encoding
        pos_emb = self.relative_positional_encoding(qlen, klen, bsz=bsz)

        pos_emb = self.dropout(pos_emb)

        # calculation
        attentions = [] if output_attentions else None
        representation = [] if self.return_representation else None
        for i, (attn, ff) in enumerate(zip(self.attn_layers, self.ff_layers)):

            outputs = attn(
                output,
                attn_mask,
                pos_emb,
                output_attentions=output_attentions,
            )

            output = ff(outputs[0])
            if self.return_representation:
                representation.append(self.dropout(output))
            if output_attentions:
                attentions.append(outputs[1])
        if output_attentions:
            attentions = ops.concat(list(map(lambda x:x.expand_dims(0), attentions)), axis=0)
        output = self.dropout(output)

        # Prepare outputs, we transpose back here to shape [bsz, len, hidden_dim] (cf. beginning of forward() method)
        output = output.transpose(1, 0, 2)

        outputs = (output,)

        if output_attentions:
            outputs = outputs + (attentions,)
        if self.return_representation and self.training:
            return outputs, representation
        return outputs
