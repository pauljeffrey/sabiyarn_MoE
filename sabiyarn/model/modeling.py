"""
SabiYarn causal LM with optional MoE layers.
Compatible with transformers 4.x–5.x (legacy and Cache KV formats).
"""

from transformers import PreTrainedModel, AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerationMixin

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None

try:
    from .configuration import GPTJXMoEConfig
except ImportError:
    from configuration import GPTJXMoEConfig

from typing import List, Optional
import math

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# KV-cache helpers (transformers 4.x tuples vs 5.x Cache objects)
# ---------------------------------------------------------------------------

def _get_past_len(past_key_values) -> int:
    if past_key_values is None:
        return 0
    if Cache is not None and isinstance(past_key_values, Cache):
        return past_key_values.get_seq_length()
    if past_key_values and past_key_values[0][0] is not None:
        return past_key_values[0][0].size(2)
    return 0


def _cache_to_legacy(past_key_values):
    if past_key_values is None:
        return None
    if Cache is not None and isinstance(past_key_values, Cache):
        if past_key_values.get_seq_length() == 0:
            return None
        if hasattr(past_key_values, "to_legacy_cache"):
            return past_key_values.to_legacy_cache()
        return tuple((layer.keys, layer.values) for layer in past_key_values.layers)
    if past_key_values and past_key_values[0][0] is None:
        return None
    return past_key_values


def _legacy_to_cache(legacy_cache):
    if legacy_cache is None:
        return None
    if Cache is not None:
        from transformers.cache_utils import DynamicCache
        if hasattr(DynamicCache, "from_legacy_cache"):
            return DynamicCache.from_legacy_cache(legacy_cache)
        return DynamicCache(ddp_cache_data=legacy_cache)
    return tuple(legacy_cache)


def _cache_is_warm(past_key_values) -> bool:
    return _get_past_len(past_key_values) > 0


def _layer_past(legacy_cache, layer_idx):
    if legacy_cache is None or layer_idx >= len(legacy_cache):
        return None
    k, v = legacy_cache[layer_idx]
    return None if k is None or v is None else (k, v)


def _expand_attn_mask(attn_mask, query_len, total_len, n_heads, has_past, device):
    """Normalize 2D/4D masks to (B, nh, T, total_len) bool for attention."""
    if attn_mask is None:
        return None

    attn_mask = attn_mask.to(torch.bool)
    if attn_mask.dim() == 2:
        b, seq = attn_mask.size()
        if seq == query_len and has_past:
            past_len = total_len - query_len
            past_mask = torch.ones(b, past_len, device=device, dtype=torch.bool)
            attn_mask = torch.cat([past_mask, attn_mask], dim=1)
        elif seq not in (query_len, total_len):
            raise ValueError(
                f"Unsupported attention_mask shape {attn_mask.shape}; "
                f"expected (B, {query_len}) or (B, {total_len})"
            )
        attn_mask = attn_mask.view(b, 1, 1, total_len).expand(b, 1, query_len, total_len)
        attn_mask = attn_mask.expand(-1, n_heads, -1, -1)
    elif attn_mask.dim() == 4:
        if attn_mask.size(-2) != query_len:
            attn_mask = attn_mask[..., -query_len:, :]
        if attn_mask.size(1) == 1:
            attn_mask = attn_mask.expand(-1, n_heads, -1, -1)
        elif attn_mask.size(1) != n_heads:
            raise ValueError(f"Mask heads {attn_mask.size(1)} != n_heads {n_heads}")
    else:
        raise ValueError(f"Unsupported attention_mask dim {attn_mask.dim()}; expected 2 or 4")
    return attn_mask


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_heads == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_heads = config.n_heads
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_heads
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x, attn_mask=None, past_key_value=None, use_cache=False):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        total_len = k.size(2)
        has_past = past_key_value is not None
        mask = _expand_attn_mask(attn_mask, T, total_len, self.n_heads, has_past, x.device)

        if self.flash:
            if mask is not None:
                y = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask,
                    dropout_p=self.dropout if self.training else 0, is_causal=False,
                )
            elif not has_past:
                y = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=None,
                    dropout_p=self.dropout if self.training else 0, is_causal=True,
                )
            else:
                causal = torch.tril(torch.ones(T, total_len, device=x.device, dtype=torch.bool))
                y = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=causal.view(1, 1, T, total_len),
                    dropout_p=self.dropout if self.training else 0, is_causal=False,
                )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
            if mask is not None:
                att = att.masked_fill(~mask, float("-inf"))
            else:
                causal = torch.tril(torch.ones(T, total_len, device=x.device, dtype=torch.bool))
                att = att.masked_fill(~causal.view(1, 1, T, total_len), float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return (y, (k.detach(), v.detach())) if use_cache else y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class BlockJ(nn.Module):

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.j = LayerNorm(config.n_embd, config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        
        # Use MoE if configured, otherwise use dense MLP
        if getattr(config, 'use_moe', False):
            self.mlp = MoE(
                num_experts_per_tok=config.num_experts_per_tok,
                num_experts=config.expert_count_for_layer(layer_idx),
                emb_dim=config.n_embd,
                moe_dim=config.moe_dim,
                dropout=config.dropout
            )
            self.use_moe = True
        else:
            self.mlp = MLP(config)
            self.use_moe = False

    def forward(self, x, attn_mask=None, past_key_value=None, use_cache=False):
        h = x
        x_ln = self.ln_1(x)
        if use_cache:
            attn_out, new_past = self.attn(x_ln, attn_mask=attn_mask, past_key_value=past_key_value, use_cache=True)
        else:
            attn_out = self.attn(x_ln, attn_mask=attn_mask, past_key_value=past_key_value, use_cache=False)
            new_past = None
        x = h + attn_out + self.j(x_ln)
        x = x + self.mlp(self.ln_2(x))
        return (x, new_past) if use_cache else x
        

class MoE(nn.Module):
    """Mixture-of-experts feed-forward block with top-k routing and GELU MLP experts."""

    def __init__(self, num_experts_per_tok: int, num_experts: int, emb_dim: int, moe_dim: int, dropout: float = 0.0, dtype=torch.float32):
        super().__init__()
        self.k = int(num_experts_per_tok)
        self.E = int(num_experts)
        self.D = int(emb_dim)
        self.H = int(moe_dim)
        self.dropout = dropout

        self.gate = nn.Linear(self.D, self.E, bias=False, dtype=dtype) # use gate variable bcause couldnt load from checkpoint
        # Match MLP structure: c_fc -> GELU -> c_proj
        self.fc_bank = nn.Parameter(torch.empty(self.E, self.D, self.H, dtype=dtype))  # Equivalent to c_fc: (n_embd -> 4*n_embd)
        self.proj_bank = nn.Parameter(torch.empty(self.E, self.H, self.D, dtype=dtype))  # Equivalent to c_proj: (4*n_embd -> n_embd)
        self.gelu = nn.GELU()  # Match MLP activation
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        # Initialize parameters
        self._init_parameters()


    def expert_utilization(self, logits):
        """Compute per-expert load and auxiliary load-balancing loss for training."""
       
        _, selected = logits.topk(self.k, dim=-1)
        selected = F.one_hot(selected, num_classes=self.E).sum(dim=2) # B, T, E

        load = torch.mean(selected.float(), dim=(0,1))
        
        # average router probability per expert
        P = torch.softmax(logits, dim=-1).float().mean(dim=(0,1))  # [E]
        self._router_probs = P.detach() # per-expert avg prob
        self._aux_lb = self.E * torch.sum(load * P)

        
        self._expert_utilization = load

    def _init_parameters(self):
        """Initialize MoE parameters following standard practices."""
        # Initialize gate with small values to start with uniform routing
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.02)
        
        # Initialize expert banks to match MLP initialization
        # fc_bank: standard normal (like c_fc in MLP)
        nn.init.normal_(self.fc_bank, mean=0.0, std=0.02)
        
        # proj_bank: smaller initialization for stability (like c_proj in MLP)
        nn.init.normal_(self.proj_bank, mean=0.0, std=0.02 / math.sqrt(2))

    def forward(self, x):
        B, T, D = x.shape
        assert D == self.D, f"Expected emb_dim={self.D}, got {D}"

        logits = self.gate(x) # B, T, E

        if self.training:
            logits = logits + torch.randn_like(logits) * 1e-1

        
        topk_logits, selected = logits.topk(self.k, dim=-1)
        topk_probs = F.softmax(topk_logits, dim=-1)

        # Match MLP structure exactly: c_fc -> GELU -> c_proj
        # Step 1: c_fc equivalent: x @ fc_bank -> (B, T, E, H)
        h = torch.einsum("btd,edh->bteh", x, self.fc_bank)  # B, T, E, H
        
        # Step 2: GELU activation (matching MLP)
        h = self.gelu(h)  # B, T, E, H
        
        # Step 3: c_proj equivalent: h @ proj_bank -> (B, T, E, D)
        y = torch.einsum("bteh,ehd->bted", h, self.proj_bank)  # B, T, E, D
        
        # Step 4: Select top-k experts and combine
        gather_idx = selected.view(B, T, -1, 1).expand(-1, -1, -1, self.D)  # B, T, K, D
        y = torch.gather(y, dim=2, index=gather_idx)  # B, T, K, D
        
        # Step 5: Weighted sum of selected experts
        y = (y * topk_probs.unsqueeze(-1)).sum(dim=2)  # B, T, D
        
        # Step 6: Apply dropout like MLP
        y = self.dropout_layer(y)

        self.expert_utilization(logits)
        return y     

    @torch.no_grad()
    def truncate_experts(self, keep: List[int]) -> int:
        """Retain only the listed experts, re-slicing gate weights and expert banks in place.

        Args:
            keep: Expert indices to keep from the current ``self.E`` experts. Order is
                preserved and becomes the new expert ordering.

        Returns:
            The new expert count stored on ``self.E``.

        Raises:
            ValueError: If ``keep`` is empty, duplicated, out of range, or smaller than
                ``num_experts_per_tok``.
        """
        keep = [int(x) for x in keep]
        E0 = int(self.E)
        k = int(self.k)
        if not keep:
            raise ValueError("keep must be non-empty")
        if len(set(keep)) != len(keep):
            raise ValueError("expert indices must be unique")
        if not set(keep).issubset(range(E0)):
            raise ValueError(f"Invalid expert indices {keep} for E={E0}")
        new_E = len(keep)
        if k > new_E:
            raise ValueError(f"num_experts_per_tok={k} > remaining experts {new_E}")

        device = self.gate.weight.device
        dtype = self.gate.weight.dtype
        keep_t = torch.tensor(keep, device=device, dtype=torch.long)
        D = int(self.D)

        gate_w = self.gate.weight.data.index_select(0, keep_t)
        self.gate = nn.Linear(D, new_E, bias=False, device=device, dtype=dtype)
        self.gate.weight.data.copy_(gate_w)

        self.fc_bank = nn.Parameter(self.fc_bank.data.index_select(0, keep_t).contiguous())
        self.proj_bank = nn.Parameter(self.proj_bank.data.index_select(0, keep_t).contiguous())
        self.E = new_E
        return new_E


class GPTJXMoEForCausalLM(PreTrainedModel, GenerationMixin):
    """SabiYarn causal language model with optional per-layer MoE feed-forward blocks."""

    config_class = GPTJXMoEConfig
    base_model_prefix = "transformer"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["BlockJ"]
    _supports_flash_attn_2 = True
    _tied_weights_keys = {"lm_head.weight": "transformer.wte.weight"}
    _keys_to_ignore_on_load_missing = [r"lm_head\.weight"]

    def __init__(self, config):
        config = self._prepare_config(config)
        super().__init__(config)
        self._all_tied_weights_keys = self._build_tied_weights_keys()

        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([BlockJ(config, layer_idx=i) for i in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        self._finalize_model_init()

    @classmethod
    def _prepare_config(cls, config: GPTJXMoEConfig) -> GPTJXMoEConfig:
        if not getattr(config, "use_moe", False) or config.expert_per_layer:
            return config
        raw = config.to_dict().get("expert_per_layer")
        if raw:
            config.expert_per_layer = {str(k): int(v) for k, v in raw.items()}
        else:
            config.expert_per_layer = {str(i): int(config.num_experts) for i in range(config.n_layer)}
        return config

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Resolve MoE layout from config/checkpoint, then load weights."""
        config = kwargs.get("config")
        token = kwargs.get("token")
        revision = kwargs.get("revision")

        if config is None:
            config = GPTJXMoEConfig.from_pretrained(
                pretrained_model_name_or_path,
                token=token,
                revision=revision,
                trust_remote_code=kwargs.get("trust_remote_code", True),
            )

        if isinstance(config, tuple):
            config = config[0]

        config = cls._prepare_config(config)
        kwargs["config"] = config

        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model._tie_word_embeddings()
        return model

    @property
    def all_tied_weights_keys(self) -> dict:
        if getattr(self, "_all_tied_weights_keys", None) is None:
            self._all_tied_weights_keys = self._build_tied_weights_keys()
        return self._all_tied_weights_keys

    @all_tied_weights_keys.setter
    def all_tied_weights_keys(self, value: dict) -> None:
        self._all_tied_weights_keys = value

    def tie_weights(self, recompute_mapping: bool = True, **kwargs):
        if not getattr(self.config, "tie_word_embeddings", True):
            self.all_tied_weights_keys = {}
            return
        self._tie_word_embeddings()
        if recompute_mapping or not getattr(self, "_all_tied_weights_keys", None):
            expand = getattr(super(), "get_expanded_tied_weights_keys", None)
            if callable(expand):
                try:
                    self.all_tied_weights_keys = expand(all_submodels=False)
                    return
                except Exception:
                    pass
            self.all_tied_weights_keys = self._build_tied_weights_keys()

    def mark_tied_weights_as_initialized(self, loading_info=None):
        for key in self.all_tied_weights_keys:
            try:
                setattr(self.get_parameter(key), "_is_hf_initialized", True)
            except (AttributeError, ValueError):
                pass
        if loading_info is None or not getattr(self, "is_custom_code", lambda: True)():
            return
        tied = self.all_tied_weights_keys
        loading_info.missing_keys = {
            key for key in loading_info.missing_keys
            if key in tied or not self._param_initialized(key)
        }

    def _param_initialized(self, key: str) -> bool:
        try:
            return bool(getattr(self.get_parameter_or_buffer(key), "_is_hf_initialized", False))
        except (AttributeError, ValueError):
            return False

    def _build_tied_weights_keys(self) -> dict:
        return dict(self._tied_weights_keys) if getattr(self.config, "tie_word_embeddings", True) else {}

    def _tie_word_embeddings(self) -> None:
        if getattr(self.config, "tie_word_embeddings", True):
            self.lm_head.weight = self.transformer.wte.weight

    def _finalize_model_init(self) -> None:
        post_init = getattr(super(), "post_init", None)
        (post_init if callable(post_init) else self.tie_weights)()

    def get_num_params(self, non_embedding=True):
        """Return total parameter count, optionally excluding position embeddings."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def get_expert_utilization(self):
        """Return per-layer expert utilization tensors and mean load-balancing loss."""
        if not getattr(self.config, 'use_moe', False):
            return None, None
        
        lb_loss, expert_utilization_per_layer = 0, []
        moe_layers = 0
        for block in self.transformer.h:
            if hasattr(block, 'use_moe') and block.use_moe and hasattr(block.mlp, '_aux_lb'):
                lb_loss += block.mlp._aux_lb
                expert_utilization_per_layer.append(block.mlp._expert_utilization.detach().cpu())
                moe_layers += 1
        
        if moe_layers > 0:
            lb_loss = lb_loss / moe_layers
        return expert_utilization_per_layer, lb_loss

    def get_input_embeddings(self):
        return self.transformer.wte

    def set_input_embeddings(self, new_embeddings):
        self.transformer.wte = new_embeddings
        self._tie_word_embeddings()

    def forward(
        self,
        input_ids,
        targets=None,
        attn_mask=None,
        attention_mask=None,
        past_key_values=None,
        position_ids=None,
        use_cache=None,
        output_hidden_states: Optional[bool] = None,
        compute_logits: bool = True,
        **kwargs
    ):
        device = input_ids.device
        b, t = input_ids.size()
        attn_mask = attention_mask if attention_mask is not None else attn_mask
        use_kv_cache = use_cache if use_cache is not None else getattr(self.config, "use_kv_cache", False)

        past_len = _get_past_len(past_key_values)
        legacy_cache = _cache_to_legacy(past_key_values)

        if position_ids is None:
            pos = torch.arange(past_len, past_len + t, dtype=torch.long, device=device)
        else:
            pos = position_ids

        total_len = past_len + t
        if total_len > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {total_len}, block size is {self.config.block_size}"
            )

        pos_1d = pos[0] if pos.dim() == 2 else pos
        pos_emb = self.transformer.wpe(pos_1d)
        if pos_emb.dim() == 2:
            pos_emb = pos_emb.unsqueeze(0).expand(b, -1, -1)

        x = self.transformer.drop(self.transformer.wte(input_ids) + pos_emb)

        new_past_key_values = [] if use_kv_cache else None
        for i, block in enumerate(self.transformer.h):
            layer_past = _layer_past(legacy_cache, i)
            if use_kv_cache:
                x, new_past = block(x, attn_mask=attn_mask, past_key_value=layer_past, use_cache=True)
                new_past_key_values.append(new_past)
            else:
                x = block(x, attn_mask=attn_mask, past_key_value=layer_past, use_cache=False)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        elif not compute_logits:
            # Caller only wants hidden_states (e.g. external cut-cross-entropy loss);
            # skip the lm_head projection entirely to avoid materializing full vocab logits.
            logits = None
            loss = None
        else:
            logits = self.lm_head(x[:, [-1], :]) if use_kv_cache and past_len > 0 else self.lm_head(x)
            loss = None

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=_legacy_to_cache(tuple(new_past_key_values)) if use_kv_cache else None,
            hidden_states=x if output_hidden_states else None,
            attentions=None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        position_ids=None,
        use_cache=None,
        **kwargs,
    ):
        use_kv_cache = use_cache if use_cache is not None else getattr(self.config, "use_kv_cache", False)
        has_cache = use_kv_cache and _cache_is_warm(past_key_values)

        model_inputs = {"input_ids": input_ids[:, -1:] if has_cache else input_ids}
        if has_cache:
            model_inputs["past_key_values"] = past_key_values
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        if use_cache is not None:
            model_inputs["use_cache"] = use_cache

        if position_ids is not None:
            if has_cache:
                position_ids = position_ids[:, -1].unsqueeze(-1)
            model_inputs["position_ids"] = position_ids
        elif has_cache:
            model_inputs["position_ids"] = torch.tensor(
                [[_get_past_len(past_key_values)]], device=input_ids.device, dtype=torch.long
            )

        for k, v in kwargs.items():
            if v is not None:
                model_inputs[k] = v
        return model_inputs

    def _reorder_cache(self, past_key_values, beam_idx: torch.Tensor):
        if Cache is not None and isinstance(past_key_values, Cache):
            past_key_values.reorder_cache(beam_idx)
            return past_key_values
        return [
            (k.index_select(0, beam_idx.to(k.device)), v.index_select(0, beam_idx.to(v.device)))
            for k, v in past_key_values
        ]


    def crop_block_size(self, block_size):
        """Reduce the maximum sequence length and trim position embeddings in place."""
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @torch.no_grad()
    def apply_moe_layer_edits(
        self,
        *,
        expert_indices_all_layers: Optional[List[int]] = None,
        expert_keep_by_layer: Optional[dict] = None,
        zero_router_for_layers=frozenset(),
        verbose: bool = True,
    ) -> None:
        """Truncate MoE experts in place and sync ``config.expert_per_layer``.

        Call this after loading a full checkpoint to select which experts each layer
        keeps. ``config.num_experts`` is not modified; only ``expert_per_layer`` is
        updated with the resulting per-layer expert counts.

        Args:
            expert_indices_all_layers: Expert indices to keep in every MoE block,
                e.g. ``[0, 1, 2, 3]``. Mutually exclusive with ``expert_keep_by_layer``.
            expert_keep_by_layer: Per-layer expert indices to keep,
                e.g. ``{0: [1, 3], 1: [3, 4, 6]}``. Unlisted MoE layers are unchanged.
            zero_router_for_layers: Layer indices whose router gates are zeroed after
                truncation to produce uniform routing.
            verbose: Whether to print a short summary.

        Raises:
            ValueError: If both truncation modes are supplied, or a target layer is
                not an MoE block.
        """
        if expert_indices_all_layers is not None and expert_keep_by_layer is not None:
            raise ValueError("Set only one of expert_indices_all_layers or expert_keep_by_layer")

        zero_router_for_layers = frozenset(int(x) for x in zero_router_for_layers)

        if expert_indices_all_layers is not None:
            keep = [int(x) for x in expert_indices_all_layers]
            for block in self.transformer.h:
                if getattr(block, "use_moe", False):
                    block.mlp.truncate_experts(keep)
            if verbose:
                print(f"MoE: all layers kept {keep}")

        elif expert_keep_by_layer:
            for li, keep in expert_keep_by_layer.items():
                block = self.transformer.h[int(li)]
                if not getattr(block, "use_moe", False):
                    raise ValueError(f"Layer {li} is not MoE")
                block.mlp.truncate_experts([int(x) for x in keep])
            if verbose:
                print("MoE: applied per-layer expert selection")

        for li in zero_router_for_layers:
            block = self.transformer.h[int(li)]
            if not getattr(block, "use_moe", False):
                raise ValueError(f"Layer {li} is not MoE")
            block.mlp.gate.weight.data.zero_()
        if zero_router_for_layers and verbose:
            print("MoE: zeroed gate at layers", sorted(zero_router_for_layers))

        if expert_indices_all_layers is not None or expert_keep_by_layer is not None:
            self._sync_expert_per_layer()

    def _sync_expert_per_layer(self) -> None:
        """Write each MoE block's live expert count into ``config.expert_per_layer``."""
        expert_per_layer = {}
        for block in self.transformer.h:
            if getattr(block, "use_moe", False):
                expert_per_layer[str(block.layer_idx)] = int(block.mlp.E)
        self.config.expert_per_layer = expert_per_layer

    def load_dense_weights_into_moe(self, dense_state_dict, strict=False):
        """
        Migrate Dense MLP weights to MoE experts.
        Ensures exact mathematical equivalence by cloning weights/biases to ALL experts.
        """
        if not getattr(self.config, 'use_moe', False):
            return self.load_state_dict(dense_state_dict, strict=strict)
        
        print("Converting Dense Checkpoint -> MoE Checkpoint...")
        moe_state_dict = {}
        
        # Get config details
        num_experts = self.config.num_experts
        moe_dim = self.config.moe_dim
        
        for key, value in dense_state_dict.items():
            # Identify MLP weights
            if 'mlp.c_fc' in key or 'mlp.c_proj' in key:
                
                # Extract layer index and type (weight/bias)
                # key format: transformer.h.{i}.mlp.c_fc.{weight/bias}
                parts = key.split('.')
                layer_idx = parts[2]
                layer_key_prefix = f"transformer.h.{layer_idx}.mlp"
                
                is_bias = 'bias' in key
                is_fc = 'c_fc' in key
                
                # --- Handle c_fc (Input -> Hidden) ---
                if is_fc:
                    if not is_bias:
                        # Weight: Dense is (H, D) -> MoE needs (E, D, H)
                        # 1. Transpose to (D, H)
                        w_T = value.t()
                        # 2. Slice to moe_dim if necessary
                        w_T = w_T[:, :moe_dim]
                        # 3. Expand to (E, D, H)
                        new_val = w_T.unsqueeze(0).expand(num_experts, -1, -1).clone()
                        moe_state_dict[f"{layer_key_prefix}.fc_bank"] = new_val
                    else:
                        # Bias: Dense is (H) -> MoE needs (E, H)
                        b = value[:moe_dim]
                        new_val = b.unsqueeze(0).expand(num_experts, -1).clone()
                        moe_state_dict[f"{layer_key_prefix}.fc_bias"] = new_val

                # --- Handle c_proj (Hidden -> Output) ---
                else: 
                    if not is_bias:
                        # Weight: Dense is (D, H) -> MoE needs (E, H, D)
                        # 1. Transpose to (H, D)
                        w_T = value.t()
                        # 2. Slice source dimension (H) if necessary
                        w_T = w_T[:moe_dim, :]
                        # 3. Expand to (E, H, D)
                        new_val = w_T.unsqueeze(0).expand(num_experts, -1, -1).clone()
                        moe_state_dict[f"{layer_key_prefix}.proj_bank"] = new_val
                    else:
                        # Bias: Dense is (D) -> MoE needs (E, D)
                        # Bias is on the output, so dimension is D, usually doesn't need slicing
                        new_val = value.unsqueeze(0).expand(num_experts, -1).clone()
                        moe_state_dict[f"{layer_key_prefix}.proj_bias"] = new_val

                # --- Initialize Gate (if not yet initialized) ---
                # We initialize gate to zero to ensure uniform routing probability initially,
                # which guarantees average of identical experts == single expert.
                gate_key = f"{layer_key_prefix}.gate.weight"
                if gate_key not in moe_state_dict:
                    # Zeros = equal probability for all experts
                    moe_state_dict[gate_key] = torch.zeros(num_experts, self.config.n_embd)

            else:
                # Copy non-MLP keys directly (Attn, LayerNorm, Embeddings)
                moe_state_dict[key] = value

        print("Loading constructed state dict...")
        return self.load_state_dict(moe_state_dict, strict=strict)
                

AutoConfig.register("sabiyarn", GPTJXMoEConfig)
AutoModel.register(GPTJXMoEConfig,GPTJXMoEForCausalLM)
AutoModelForCausalLM.register(GPTJXMoEConfig, GPTJXMoEForCausalLM)   