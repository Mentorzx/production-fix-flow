import numpy as np


def test_random_baselines():
    num_entities = 21966

    expected_mrr = np.log(num_entities) / num_entities
    expected_hits10 = 10.0 / num_entities
    expected_mean_rank = (num_entities + 1) / 2.0

    print(f"\n[BASELINE] N={num_entities}")
    print(f"  Random MRR: {expected_mrr:.6f}")
    print(f"  Random Hits@10: {expected_hits10:.6f}")
    print(f"  Random Mean Rank: {expected_mean_rank:.1f}")

    # Check current reported values
    observed_mrr = 0.0020
    observed_hits10 = 0.0002

    ratio_mrr = observed_mrr / expected_mrr
    print(f"  Observed MRR / Random: {ratio_mrr:.2f}x (Better than random)")

    ratio_hits = observed_hits10 / expected_hits10
    print(f"  Observed Hits / Random: {ratio_hits:.2f}x (Worse than random?)")

    # If Hits@10 is worse than random but MRR is better, it implies:
    # The model puts the true answer in the top ranks (e.g. top 100) more often than random,
    # boosting MRR (1/rank), but rarely in the top 10.
    # OR: The distribution of ranks is heavy-tailed differently.

    # 1/100 = 0.01. 1/1000 = 0.001.
    # If we have many ranks around 500, MRR ~ 0.002.
    # Hits@10 would be 0.

    # This confirms the model IS learning something (better than random MRR),
    # but it's not precise enough to hit top 10 yet.


if __name__ == "__main__":
    test_random_baselines()
