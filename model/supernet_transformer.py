import paddle
import math
from model.module.Linear_super import LinearSuper
from model.module.layernorm_super import LayerNormSuper
from model.module.multihead_super import AttentionSuper
from model.module.embedding_super import PatchembedSuper
from model.utils import trunc_normal_
from model.utils import DropPath
import numpy as np


def gelu(x: paddle.Tensor) ->paddle.Tensor:
    if hasattr(paddle.nn.functional, 'gelu'):
        return paddle.nn.functional.gelu(x=x.astype(dtype='float32')).astype(
            dtype=x.dtype)
    else:
        return x * 0.5 * (1.0 + paddle.erf(x=x / math.sqrt(2.0)))


class Vision_TransformerSuper(paddle.nn.Layer):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes
        =1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
        qkv_bias=False, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0,
        drop_path_rate=0.0, pre_norm=True, scale=False, gp=False,
        relative_position=False, change_qkv=False, abs_pos=True,
        max_relative_position=14):
        super(Vision_TransformerSuper, self).__init__()
        self.super_embed_dim = embed_dim
        self.super_mlp_ratio = mlp_ratio
        self.super_layer_num = depth
        self.super_num_heads = num_heads
        self.super_dropout = drop_rate
        self.super_attn_dropout = attn_drop_rate
        self.num_classes = num_classes
        self.pre_norm = pre_norm
        self.scale = scale
        self.patch_embed_super = PatchembedSuper(img_size=img_size,
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.gp = gp
        self.sample_embed_dim = None
        self.sample_mlp_ratio = None
        self.sample_layer_num = None
        self.sample_num_heads = None
        self.sample_dropout = None
        self.sample_output_dim = None
        self.blocks = paddle.nn.LayerList()
        dpr = [x.item() for x in paddle.linspace(start=0, stop=
            drop_path_rate, num=depth)]
        for i in range(depth):
            self.blocks.append(TransformerEncoderLayer(dim=embed_dim,
                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale, dropout=drop_rate, attn_drop=
                attn_drop_rate, drop_path=dpr[i], pre_norm=pre_norm, scale=
                self.scale, change_qkv=change_qkv, relative_position=
                relative_position, max_relative_position=max_relative_position)
                )
        num_patches = self.patch_embed_super.num_patches
        self.abs_pos = abs_pos
        if self.abs_pos:
            out_2 = paddle.create_parameter(shape=paddle.zeros(shape=[1, 
                num_patches + 1, embed_dim]).shape, dtype=paddle.zeros(
                shape=[1, num_patches + 1, embed_dim]).numpy().dtype,
                default_initializer=paddle.nn.initializer.Assign(paddle.
                zeros(shape=[1, num_patches + 1, embed_dim])))
            out_2.stop_gradient = not True
            self.pos_embed = out_2
            trunc_normal_(self.pos_embed, std=0.02)
        out_3 = paddle.create_parameter(shape=paddle.zeros(shape=[1, 1,
            embed_dim]).shape, dtype=paddle.zeros(shape=[1, 1, embed_dim]).
            numpy().dtype, default_initializer=paddle.nn.initializer.Assign
            (paddle.zeros(shape=[1, 1, embed_dim])))
        out_3.stop_gradient = not True
        self.cls_token = out_3
        trunc_normal_(self.cls_token, std=0.02)
        if self.pre_norm:
            self.norm = LayerNormSuper(super_embed_dim=embed_dim)
        self.head = LinearSuper(embed_dim, num_classes
            ) if num_classes > 0 else paddle.nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, paddle.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, paddle.nn.Linear) and m.bias is not None:
                init_Constant = paddle.nn.initializer.Constant(value=0)
                init_Constant(m.bias)
        elif isinstance(m, paddle.nn.LayerNorm):
            init_Constant = paddle.nn.initializer.Constant(value=0)
            init_Constant(m.bias)
            init_Constant = paddle.nn.initializer.Constant(value=1.0)
            init_Constant(m.weight)


    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'rel_pos_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = paddle.nn.Linear(in_features=self.embed_dim,
            out_features=num_classes
            ) if num_classes > 0 else paddle.nn.Identity()

    def set_sample_config(self, config: dict):
        self.sample_embed_dim = config['embed_dim']
        self.sample_mlp_ratio = config['mlp_ratio']
        self.sample_layer_num = config['layer_num']
        self.sample_num_heads = config['num_heads']
        self.sample_dropout = calc_dropout(self.super_dropout, self.
            sample_embed_dim[0], self.super_embed_dim)
        self.patch_embed_super.set_sample_config(self.sample_embed_dim[0])
        self.sample_output_dim = [out_dim for out_dim in self.
            sample_embed_dim[1:]] + [self.sample_embed_dim[-1]]
        for i, blocks in enumerate(self.blocks):
            if i < self.sample_layer_num:
                sample_dropout = calc_dropout(self.super_dropout, self.
                    sample_embed_dim[i], self.super_embed_dim)
                sample_attn_dropout = calc_dropout(self.super_attn_dropout,
                    self.sample_embed_dim[i], self.super_embed_dim)
                blocks.set_sample_config(is_identity_layer=False,
                    sample_embed_dim=self.sample_embed_dim[i],
                    sample_mlp_ratio=self.sample_mlp_ratio[i],
                    sample_num_heads=self.sample_num_heads[i],
                    sample_dropout=sample_dropout, sample_out_dim=self.
                    sample_output_dim[i], sample_attn_dropout=
                    sample_attn_dropout)
            else:
                blocks.set_sample_config(is_identity_layer=True)
        if self.pre_norm:
            self.norm.set_sample_config(self.sample_embed_dim[-1])
        self.head.set_sample_config(self.sample_embed_dim[-1], self.num_classes
            )

    def get_sampled_params_numel(self, config):
        self.set_sample_config(config)
        numels = []
        for name, module in self.named_sublayers():
            if hasattr(module, 'calc_sampled_param_num'):
                if name.split('.')[0] == 'blocks' and int(name.split('.')[1]
                    ) >= config['layer_num']:
                    continue
                numels.append(module.calc_sampled_param_num())
        return sum(numels) + self.sample_embed_dim[0] * (2 + self.
            patch_embed_super.num_patches)

    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += self.patch_embed_super.get_complexity(sequence_length)
        total_flops += np.prod(tuple(self.pos_embed[(...), :self.
            sample_embed_dim[0]].shape)) / 2.0
        for blk in self.blocks:
            total_flops += blk.get_complexity(sequence_length + 1)
        total_flops += self.head.get_complexity(sequence_length + 1)
        return total_flops

    def forward_features(self, x):
        B = tuple(x.shape)[0]
        x = self.patch_embed_super(x)
        cls_tokens = self.cls_token[(...), :self.sample_embed_dim[0]].expand(
            shape=[B, -1, -1])
        x = paddle.concat(x=(cls_tokens, x), axis=1)
        if self.abs_pos:
            x = x + self.pos_embed[(...), :self.sample_embed_dim[0]]
        x = paddle.nn.functional.dropout(x=x, p=self.sample_dropout,
            training=self.training)
        for blk in self.blocks:
            x = blk(x)
        if self.pre_norm:
            x = self.norm(x)
        if self.gp:
            return paddle.mean(x=x[:, 1:], axis=1)
        return x[:, (0)]

    def forward(self, x):
#        print("supernet——transformer中的forward中的x是"+x)
        x = self.forward_features(x)
        x = self.head(x)

        return x


class TransformerEncoderLayer(paddle.nn.Layer):
    """Encoder layer block.

    Args:
        args (argparse.Namespace): parsed command-line arguments which
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=False,
        qk_scale=None, dropout=0.0, attn_drop=0.0, drop_path=0.0, act_layer
        =paddle.nn.GELU, pre_norm=True, scale=False, relative_position=
        False, change_qkv=False, max_relative_position=14):
        super().__init__()
        self.super_embed_dim = dim
        self.super_mlp_ratio = mlp_ratio
        self.super_ffn_embed_dim_this_layer = int(mlp_ratio * dim)
        self.super_num_heads = num_heads
        self.normalize_before = pre_norm
        self.super_dropout = attn_drop
        self.drop_path = DropPath(drop_path
            ) if drop_path > 0.0 else paddle.nn.Identity()
        self.scale = scale
        self.relative_position = relative_position
        self.sample_embed_dim = None
        self.sample_mlp_ratio = None
        self.sample_ffn_embed_dim_this_layer = None
        self.sample_num_heads_this_layer = None
        self.sample_scale = None
        self.sample_dropout = None
        self.sample_attn_dropout = None
        self.is_identity_layer = None
        self.attn = AttentionSuper(dim, num_heads=num_heads, qkv_bias=
            qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=
            dropout, scale=self.scale, relative_position=self.
            relative_position, change_qkv=change_qkv, max_relative_position
            =max_relative_position)
        self.attn_layer_norm = LayerNormSuper(self.super_embed_dim)
        self.ffn_layer_norm = LayerNormSuper(self.super_embed_dim)
        self.activation_fn = gelu
        self.fc1 = LinearSuper(super_in_dim=self.super_embed_dim,
            super_out_dim=self.super_ffn_embed_dim_this_layer)
        self.fc2 = LinearSuper(super_in_dim=self.
            super_ffn_embed_dim_this_layer, super_out_dim=self.super_embed_dim)

    def set_sample_config(self, is_identity_layer, sample_embed_dim=None,
        sample_mlp_ratio=None, sample_num_heads=None, sample_dropout=None,
        sample_attn_dropout=None, sample_out_dim=None):
        if is_identity_layer:
            self.is_identity_layer = True
            return
        self.is_identity_layer = False
        self.sample_embed_dim = sample_embed_dim
        self.sample_out_dim = sample_out_dim
        self.sample_mlp_ratio = sample_mlp_ratio
        self.sample_ffn_embed_dim_this_layer = int(sample_embed_dim *
            sample_mlp_ratio)
        self.sample_num_heads_this_layer = sample_num_heads
        self.sample_dropout = sample_dropout
        self.sample_attn_dropout = sample_attn_dropout
        self.attn_layer_norm.set_sample_config(sample_embed_dim=self.
            sample_embed_dim)
        self.attn.set_sample_config(sample_q_embed_dim=self.
            sample_num_heads_this_layer * 64, sample_num_heads=self.
            sample_num_heads_this_layer, sample_in_embed_dim=self.
            sample_embed_dim)
        self.fc1.set_sample_config(sample_in_dim=self.sample_embed_dim,
            sample_out_dim=self.sample_ffn_embed_dim_this_layer)
        self.fc2.set_sample_config(sample_in_dim=self.
            sample_ffn_embed_dim_this_layer, sample_out_dim=self.sample_out_dim
            )
        self.ffn_layer_norm.set_sample_config(sample_embed_dim=self.
            sample_embed_dim)

    def forward(self, x):
        """
        Args:
            x (Tensor): input to the layer of shape `(batch, patch_num , sample_embed_dim)`

        Returns:
            encoded output of shape `(batch, patch_num, sample_embed_dim)`
        """
        if self.is_identity_layer:
            return x
        residual = x
        x = self.maybe_layer_norm(self.attn_layer_norm, x, before=True)
        x = self.attn(x)
        x = paddle.nn.functional.dropout(x=x, p=self.sample_attn_dropout,
            training=self.training)
        x = self.drop_path(x)
        x = residual + x
        x = self.maybe_layer_norm(self.attn_layer_norm, x, after=True)
        residual = x
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = paddle.nn.functional.dropout(x=x, p=self.sample_dropout,
            training=self.training)
        x = self.fc2(x)
        x = paddle.nn.functional.dropout(x=x, p=self.sample_dropout,
            training=self.training)
        if self.scale:
            x = x * (self.super_mlp_ratio / self.sample_mlp_ratio)
        x = self.drop_path(x)
        x = residual + x
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, after=True)
        return x

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def get_complexity(self, sequence_length):
        total_flops = 0
        if self.is_identity_layer:
            return total_flops
        total_flops += self.attn_layer_norm.get_complexity(sequence_length + 1)
        total_flops += self.attn.get_complexity(sequence_length + 1)
        total_flops += self.ffn_layer_norm.get_complexity(sequence_length + 1)
        total_flops += self.fc1.get_complexity(sequence_length + 1)
        total_flops += self.fc2.get_complexity(sequence_length + 1)
        return total_flops


def calc_dropout(dropout, sample_embed_dim, super_embed_dim):
    return dropout * 1.0 * sample_embed_dim / super_embed_dim
