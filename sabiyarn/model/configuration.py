from typing import Dict, Optional

from transformers import PretrainedConfig


class GPTJXMoEConfig(PretrainedConfig):
    """Configuration for SabiYarn causal language models with optional MoE layers."""

    model_type = "sabiyarn"
    attribute_map = {
        "hidden_size": "n_embd",
        "num_attention_heads": "n_heads",
        "num_hidden_layers": "n_layer",
        "max_position_embeddings": "block_size",
    }

    def __init__(        self,
        block_size: int = 32768,
        vocab_size: int = 52050,
        n_layer: int = 12,
        n_heads: int = 12,
        n_embd: int = 768,
        dropout: float = 0.0,
        max_batch_size: int = 1,
        use_kv_cache: bool = True,
        bias: bool = False,
        kv_cache_dtype: str = "float32",
        use_moe: bool = False,
        num_experts: int = 4,
        num_experts_per_tok: int = 2,
        moe_dim: Optional[int] = None,
        expert_per_layer: Optional[Dict[str, int]] = None,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_heads = n_heads
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias
        self.use_kv_cache = use_kv_cache
        self.max_batch_size = max_batch_size
        self.kv_cache_dtype = kv_cache_dtype

        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_dim = moe_dim if moe_dim is not None else (4 * n_embd)
        self.tie_word_embeddings = tie_word_embeddings
        self.expert_per_layer = (
            {str(k): int(v) for k, v in expert_per_layer.items()} if expert_per_layer else None
        )

        # HuggingFace-standard aliases used by generation/cache utilities.
        self.num_hidden_layers = n_layer
        self.hidden_size = n_embd
        self.num_attention_heads = n_heads
        self.max_position_embeddings = block_size

        super().__init__(**kwargs)

    def expert_count_for_layer(self, layer_idx: int) -> int:
        """Return expert count for a layer: ``expert_per_layer`` first, else ``num_experts``."""
        if not self.use_moe:
            return int(self.num_experts)
        if self.expert_per_layer:
            count = self.expert_per_layer.get(str(layer_idx))
            if count is not None:
                return int(count)
        return int(self.num_experts)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load config; unwrap ``(config, unused_kwargs)`` tuple on transformers >=5."""
        return_unused = kwargs.get("return_unused_kwargs", False)
        result = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        if isinstance(result, tuple):
            config, unused_kwargs = result[0], result[1] if len(result) > 1 else {}
        else:
            config, unused_kwargs = result, {}

        if return_unused:
            return config, unused_kwargs
        return config
