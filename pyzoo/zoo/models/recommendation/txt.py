from gluonnlp.model.transformer import TransformerEncoder, TransformerEncoderCell
from mxnet.gluon.block import HybridBlock


class MeanMaxPooling(HybridBlock):
    def __init__(self, axis=1, dropout=0.0, prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        self.axis = axis
        self.dropout = dropout

    def hybrid_forward(self, F, inputs):
        mean_out = F.mean(data=inputs, axis=self.axis)
        max_out = F.max(data=inputs, axis=self.axis)
        outputs = F.concat(mean_out, max_out, dim=1)
        if self.dropout:
            outputs = F.Dropout(data=outputs, p=self.dropout)
        outputs = F.LayerNorm(data=outputs)
        return outputs


class SequenceTransformer(HybridBlock):
    def __init__(self, num_items, item_embed, item_hidden_size, item_max_length, item_num_heads,
                 item_num_layers, item_transformer_dropout, item_pooling_dropout, prefix=None,
                 params=None):
        super().__init__(prefix=prefix, params=params)
        self.num_items = num_items
        self.item_embed = item_embed
        with self.name_scope():
            self.item_pooling_dp = MeanMaxPooling(dropout=item_pooling_dropout)
            self.item_encoder = TransformerEncoder(units=item_embed, hidden_size=item_hidden_size,
                                                   num_heads=item_num_heads,
                                                   num_layers=item_num_layers,
                                                   max_length=item_max_length,
                                                   dropout=item_transformer_dropout)

    def hybrid_forward(self, F, input_item, item_valid_length=None):
        item_embed_out = F.Embedding(data=input_item, input_dim=self.num_items,
                                     output_dim=self.item_embed)
        item_encoding, item_att = self.item_encoder.hybrid_forward(F, inputs=item_embed_out,
                                                                   valid_length=item_valid_length)
        item_out = self.item_pooling_dp.hybrid_forward(F, inputs=item_encoding)
        item_out = F.FullyConnected(data=item_out, num_hidden=self.cross_size)

        return item_out


class ContextTransformer(HybridBlock):
    def __init__(self, context_dims, context_embed,
                 context_hidden_size, context_max_length, context_num_heads, context_num_layers,
                 context_transformer_dropout, context_pooling_dropout, cross_size, prefix=None,
                 params=None):
        super().__init__(prefix=prefix, params=params)
        self.context_dims = context_dims
        self.context_embed = context_embed
        self.cross_size = cross_size
        with self.name_scope():
            self.context_pooling_dp = MeanMaxPooling(dropout=context_pooling_dropout)
            self.context_encoder = TransformerEncoderCell(units=context_embed,
                                                          hidden_size=context_hidden_size,
                                                          num_heads=context_num_heads,
                                                          num_layers=context_num_layers,
                                                          max_length=context_max_length,
                                                          dropout=context_transformer_dropout)

    def hybrid_forward(self, F, input_context_list):
        context_embed = [F.Embedding(data=input_context_list[i], input_dim=context_dim,
                                     output_dim=self.context_embed)
                         for i, context_dim in enumerate(self.context_dims)]
        context_input = []
        for i in context_embed:
            context_input.append(F.expand_dims(i, axis=1))
        context_embedding = F.concat(*context_input, dim=1)
        context_encoding, context_att = self.context_encoder. \
            hybrid_forward(F, inputs=context_embedding)
        context_out = self.context_pooling_dp.hybrid_forward(F, inputs=context_encoding)
        context_out = F.FullyConnected(data=context_out, num_hidden=self.cross_size)

        return context_out


class TxT(HybridBlock):
    def __init__(self, num_items, context_dims, item_embed=100, context_embed=100,
                 item_hidden_size=256, item_max_length=8, item_num_heads=4, item_num_layers=2,
                 item_transformer_dropout=0.0, item_pooling_dropout=0.1, context_hidden_size=256,
                 context_max_length=4, context_num_heads=2, context_num_layers=1,
                 context_transformer_dropout=0.0, context_pooling_dropout=0.0, act_type="relu",
                 cross_size=100, prefix=None, params=None):
        super().__init__(prefix=prefix, params=params)
        self.num_items = num_items
        self.act_type = act_type
        with self.name_scope():
            self.item_out = SequenceTransformer(num_items=num_items, item_embed=item_embed,
                                                item_hidden_size=item_hidden_size,
                                                item_max_length=item_max_length,
                                                item_num_heads=item_num_heads,
                                                item_num_layers=item_num_layers,
                                                item_transformer_dropout=item_transformer_dropout,
                                                item_pooling_dropout=item_pooling_dropout,
                                                prefix=prefix, params=params)
            self.context_out = ContextTransformer(context_dims, context_embed=context_embed,
                                                  context_hidden_size=context_hidden_size,
                                                  context_max_length=context_max_length,
                                                  context_num_heads=context_num_heads,
                                                  context_num_layers=context_num_layers,
                                                  context_transformer_dropout=
                                                  context_transformer_dropout,
                                                  context_pooling_dropout=context_pooling_dropout,
                                                  cross_size=cross_size,
                                                  prefix=prefix, params=params)

    def hybrid_forward(self, F, label, item_valid_length=None):
        outs = F.broadcast_mul(self.item_out, self.context_out)
        outs = F.Activation(data=outs, act_type=self.act_type)
        outs = F.FullyConnected(data=outs, num_hidden=int(self.num_items))
        outs = F.SoftmaxOutput(data=outs, label=label)

        return outs
