import torch

from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCConfig, DSLFMKGCModel
from pff.domain.learning.dslfm.kgc_manager import DSLFMKGCManager, KGCTrainingConfig


class MockPersistence:
    def save_checkpoint(self, data, filename):
        pass

    def load_checkpoint(self, filename, map_location=None):
        return None


def test_gradient_monitoring():
    """Monitor gradients and updates for DSLFM-KGC model."""

    config = DSLFMKGCConfig(
        num_entities=1000,
        num_relations=10,
        entity_dim=64,
        feature_dim=64,
        hidden_dim=128,
    )
    model = DSLFMKGCModel(config)

    # Setup training
    train_cfg = KGCTrainingConfig(
        batch_size=32,
        epochs=1,
        learning_rate=1e-3,
        mixed_precision=False,  # simplify debugging
    )
    manager = DSLFMKGCManager(model.config, train_cfg, persistence_port=MockPersistence())
    manager.model = model
    manager.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Create batch
    heads = torch.randint(0, 1000, (32,))
    relations = torch.randint(0, 10, (32,))
    tails = torch.randint(0, 1000, (32,))

    # Initial weights snapshot
    init_emb = model.entity_embedding.weight.clone()
    init_W = model.decoder.W.clone()

    # Forward & Backward
    model.train()
    manager.optimizer.zero_grad()
    losses = model.compute_loss(heads, relations, tails, use_inbatch_negatives=True)
    loss = losses["loss"]
    loss.backward()

    print(f"\n[DYNAMICS] Loss: {loss.item():.4f}")

    # 1. Gradient Norms (Before Clipping)
    total_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm**2
            if param_norm > 1.0:  # Report high grads
                print(f"  Grad Norm {name}: {param_norm:.4f}")
    total_norm = total_norm**0.5
    print(f"  Global Grad Norm (Before Clip): {total_norm:.4f}")

    # Check clipping config
    # KGCManager uses default clip? It's not in KGCTrainingConfig dataclass!
    # Checking code...
    # manager._optimizer_step() calls scaler.step or optimizer.step.
    # It does NOT seem to call clip_grad_norm_ explicitly in the code I read.

    # 2. Update Step
    manager.optimizer.step()

    # 3. Update Ratios
    with torch.no_grad():
        update_emb = (model.entity_embedding.weight - init_emb).norm().item()
        weight_emb = init_emb.norm().item()
        ratio_emb = update_emb / (weight_emb + 1e-9)

        update_W = (model.decoder.W - init_W).norm().item()
        weight_W = init_W.norm().item()
        ratio_W = update_W / (weight_W + 1e-9)

        print(
            f"\n[UPDATES] Entity Emb: Update={update_emb:.4f}, Weight={weight_emb:.4f}, Ratio={ratio_emb:.4e}"
        )
        print(
            f"[UPDATES] Decoder W:  Update={update_W:.4f}, Weight={weight_W:.4f}, Ratio={ratio_W:.4e}"
        )

        # Heuristic: Good update ratios are often around 1e-3 to 1e-2.
        # If < 1e-5, learning is too slow. If > 1e-1, unstable.


if __name__ == "__main__":
    test_gradient_monitoring()
