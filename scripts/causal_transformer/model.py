import os, math
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.cuda.amp import autocast
from transformers.activations import ACT2FN

class Conv1D(nn.Module):
    """
    References:
    https://github.com/huggingface/transformers/blob/v4.5.0/src/transformers/modeling_utils.py#L1201C1-L1224C17

    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (:obj:`int`): The number of output features.
        nx (:obj:`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x

"""
References:
https://github.com/huggingface/transformers/blob/v4.5.0/src/transformers/models/gpt2/modeling_gpt2.py#L125

"""
class Attention(nn.Module):
    def __init__(
        self,
        config,
        layer_idx,
        log_activations=False,
    ):

        super().__init__()

        #self.config = config
        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False) # ! -1e4 magic number is important!

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_weights = config.scale_attn_weights
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx

        self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.log_activations = log_activations
        if self.log_activations:
            self.activation_stats = {
                "attention_weight_max": None,
                "attention_weight_min": None,
            }
        

    def _attn(self, q, k, v):
        """
        Taken from:
            https://github.com/huggingface/transformers/blob/v4.5.0/src/transformers/models/gpt2/modeling_gpt2.py#L167

        We log extra statistics about the attention weights!
        """
        # @MERCURY =>> Reorder Scaled Dot-Product Attention Computation, Upcast to FP32
        # Q :: [bsz, num_heads, seq_len, dk], K :: [bsz, num_heads, dk, seq_len]
        # Get QKV Dimensions
        bsz, num_heads, seq_len, dk = q.size()

        # @MERCURY =>> Scale by SQRT(head_dim) * layer_number -- taken from Megatron LM!
        scale_factor = 1.0
        if  self.scale_attn_weights:
            scale_factor /= (float(v.size(-1)) ** 0.5)
        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= (self.layer_idx + 1)

        # Preallocate Scaled Dot-Product Tensor
        w = torch.empty(
            bsz * num_heads,
            seq_len,
            seq_len,
            dtype=q.dtype,
            device=torch.cuda.current_device(),
        )

        # Upcasting --> Disable autocast AND manually call .float()
        # Reorder via `baddbmm` Time (Scale K by 1 / root(dk) first!)
        with autocast(enabled=False):
            q, k = q.reshape(-1, seq_len, dk), k.reshape(-1, dk, seq_len)
            w = torch.baddbmm(
                w.float(),
                q.float(),
                k.float(),
                beta=0.0,
                alpha=scale_factor,
            )
            w = w.reshape(bsz, num_heads, seq_len, seq_len)


        # Add extra logging of the attention weight
        if self.log_activations:
            with torch.no_grad():
                self.activation_stats["attention_weight_max"] = w.max().item()
                self.activation_stats["attention_weight_min"] = w.min().item()

        # Apply the causal mask
        nd, ns = w.size(-2), w.size(-1)
        mask = self.bias[:, :, ns - nd : ns, :ns]
        mask_value = torch.finfo(w.dtype).min
        mask_value = torch.tensor(mask_value, dtype=w.dtype).to(w.device)
        w = torch.where(mask, w, mask_value)

        w = nn.Softmax(dim=-1)(w)

        # @MERCURY =>> Downcast (if necessary) back to V dtype (fp16 if mixed-precision)!
        # Note: This is a No-Op if Upcasting is disabled...
        w = w.type(v.dtype)

        w = self.attn_dropout(w)

        res = torch.matmul(w, v)

        return res
    
    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.num_heads, x.size(-1) // self.num_heads)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
        
    def forward(self, hidden_states,):
        query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        
        a = self._attn(query, key, value)

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        return a
    

class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        hidden_size = config.hidden_size
        mlp_inner_dim = config.mlp_dim_multipler * hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = Attention(config, layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        
        self.mlp = MLP(mlp_inner_dim, config)

    def forward(self, hidden_states):
        attn_output = self.attn(self.ln_1(hidden_states))
        hidden_states = attn_output + hidden_states # residual connection

        feed_forward_hidden_states = self.mlp(self.ln_2(hidden_states))
        hidden_states = hidden_states + feed_forward_hidden_states # residual connection

        return hidden_states  
    

class MLP(nn.Module):
    def __init__(self, mlp_inner_dim, config):  # in MLP: n_state=3072 (4 * n_embd)
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(mlp_inner_dim, embed_dim)
        self.c_proj = Conv1D(embed_dim, mlp_inner_dim)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


class Causal_Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.embed_dim = config.hidden_size
        self.vocab_size = len(config.vocab)

        self.wte = nn.Embedding(self.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

         # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name == "c_proj.weight":
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.num_hidden_layers)))


    def forward(
        self,
        input_ids,
        position_ids=None,
    ):

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if position_ids is not None: position_ids = position_ids.view(-1, input_shape[-1])

        past_length = 0
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),) # bs, seq_len, hidden_size

        for block in self.h:
            hidden_states = block(hidden_states)

        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(*output_shape)

        lm_logits = self.lm_head(hidden_states) # loss calculated outside

        return lm_logits