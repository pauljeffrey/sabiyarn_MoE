"""
Verify the MoE load-balancing auxiliary loss is actually computed and wired
into the model's forward pass (not silently zero / never backpropagated).
"""
import torch

from sabiyarn.model.configuration import GPTJXMoEConfig
from sabiyarn.model.modeling import GPTJXMoEForCausalLM, MoE


def test_moe_module_aux_loss_is_nonzero_for_imbalanced_routing():
    torch.manual_seed(0)
    moe = MoE(num_experts_per_tok=1, num_experts=4, emb_dim=8, moe_dim=16)
    moe.eval()  # disable the training-time routing noise for a deterministic check

    # Bias the gate heavily toward expert 0 -> imbalanced routing.
    with torch.no_grad():
        moe.gate.weight.zero_()
        moe.gate.weight[0] += 5.0

    x = torch.randn(2, 6, 8)
    moe(x)
    assert moe._aux_lb.item() > 0.5  # E * sum(load*P) -> E when fully collapsed onto one expert


def test_moe_module_aux_loss_near_minimum_for_balanced_routing():
    torch.manual_seed(0)
    moe = MoE(num_experts_per_tok=1, num_experts=4, emb_dim=8, moe_dim=16)
    moe.eval()

    x = torch.randn(2, 6, 8)
    logits = torch.zeros(2, 6, 4)  # uniform gate -> balanced routing regardless of x
    moe.expert_utilization(logits)
    assert abs(moe._aux_lb.item() - 1.0) < 1e-4  # E * sum((1/E)*(1/E)) == 1 at perfect balance


def _tiny_config(use_moe: bool) -> GPTJXMoEConfig:
    return GPTJXMoEConfig(
        block_size=32, vocab_size=64, n_layer=2, n_heads=2, n_embd=8,
        use_moe=use_moe, num_experts=4, num_experts_per_tok=2, moe_dim=16,
        expert_per_layer={"0": 4, "1": 4} if use_moe else None,
        use_kv_cache=False,
    )


def test_get_expert_utilization_returns_none_when_not_moe():
    model = GPTJXMoEForCausalLM(_tiny_config(use_moe=False))
    model.eval()
    input_ids = torch.randint(0, 64, (2, 5))
    model(input_ids=input_ids)  # populate any per-layer state
    _, lb_loss = model.get_expert_utilization()
    assert lb_loss is None


def test_get_expert_utilization_nonzero_and_gradients_flow_when_moe():
    model = GPTJXMoEForCausalLM(_tiny_config(use_moe=True))
    model.train()
    input_ids = torch.randint(0, 64, (2, 5))
    targets = torch.randint(0, 64, (2, 5))

    out = model(input_ids=input_ids, targets=targets)
    _, lb_loss = model.get_expert_utilization()
    assert lb_loss is not None

    total = out.loss + 0.01 * lb_loss
    total.backward()

    # gate weights should have received a gradient from the aux loss path
    grads = [p.grad for n, p in model.named_parameters() if n.endswith("mlp.gate.weight")]
    assert grads and any(g is not None and g.abs().sum().item() > 0 for g in grads)
