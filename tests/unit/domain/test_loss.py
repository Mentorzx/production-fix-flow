import torch
import torch.nn.functional as F


def test_loss_behavior():
    # Simulate scores
    # If model is learning correctly, pos_scores should increase relative to neg_scores
    pos = torch.tensor([1.0, 2.0], requires_grad=True)
    neg = torch.tensor([[0.5, 0.4], [1.5, 1.2]], requires_grad=True)
    temp = 0.5

    pos_scaled = pos / temp
    neg_scaled = neg / temp

    logits = torch.cat([pos_scaled.unsqueeze(1), neg_scaled], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long)

    loss = F.cross_entropy(logits, labels)
    print(f"Initial Loss: {loss.item()}")

    loss.backward()
    print(f"Pos grad: {pos.grad}")  # Should be negative (increase pos to decrease loss)
    print(f"Neg grad: {neg.grad}")  # Should be positive (decrease neg to decrease loss)

    # What if temp is very small?
    temp_small = 0.01
    loss_small = F.cross_entropy(
        torch.cat([(pos / temp_small).unsqueeze(1), neg / temp_small], dim=1), labels
    )
    print(f"Loss with small temp: {loss_small.item()}")


if __name__ == "__main__":
    test_loss_behavior()
