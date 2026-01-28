import numpy as np
import torch

from pff.domain.learning.dslfm.kgc_manager import DSLFMKGCConfig, DSLFMKGCManager, KGCTrainingConfig
from pff.domain.ports.persistence.model_persistence import ModelPersistencePort


def create_synthetic_transitive_data(num_entities=100, num_chains=20):
    """
    Creates a transitive dataset: A -> B, B -> C, implies A -> C.
    The model should easily learn this pattern.
    """
    triples = []

    # 0 = "implies" relation or generic connection
    # Let's use 3 relations:
    # 0: Direct connection
    # 1: Transitive connection (A->C inferred from A->B, B->C)

    for i in range(num_chains):
        a = i * 3
        b = i * 3 + 1
        c = i * 3 + 2

        if c >= num_entities:
            break

        # A -> B (r=0)
        triples.append([a, 0, b])
        # B -> C (r=0)
        triples.append([b, 0, c])
        # A -> C (r=1) - The target implication
        triples.append([a, 1, c])

    return np.array(triples)


class MockPersistence(ModelPersistencePort):
    def save_checkpoint(self, payload, filename):
        pass

    def load_checkpoint(self, filename, map_location):
        return None

    def save_model(self, model, filename):
        pass

    def load_model(self, filename, map_location):
        return None


class TestDSLFMHighScoreRepro:
    def test_dslfm_learns_transitive_pattern_high_score(self):
        """
        Golden Fixture Test:
        Verifies that DSLFM can achieve > 0.5 MCC on a simple transitive pattern.
        If this fails, the core model logic is broken.
        """
        # 1. Setup Data
        num_entities = 200
        num_relations = 2
        triples = create_synthetic_transitive_data(num_entities=num_entities, num_chains=60)

        # Fixed seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)

        indices = np.arange(len(triples))
        np.random.shuffle(indices)
        split = int(0.8 * len(triples))
        train_triples = triples[indices[:split]]
        valid_triples = triples[indices[split:]]

        # 2. Config for High Performance
        train_config = KGCTrainingConfig(
            epochs=300,
            batch_size=64,
            effective_batch_size=64,
            learning_rate=0.1,
            validate_every=10,
            early_stopping_patience=150,
            time_budget={"max_time_s": 600},
            use_compile=False,
            mixed_precision=False,
            min_kl_weight=0.0,
            max_kl_weight=0.0,
        )

        class DebugObserver:
            def on_training_start(self, config):
                print("\n[Debug] Training started")

            def on_epoch_start(self, epoch):
                pass

            def on_epoch_end(self, epoch, metrics):
                print(
                    f"[Debug] Epoch {epoch}: Loss={metrics.get('loss', 'N/A'):.4f}, MRR={metrics.get('mcc', 0.0):.4f} (real MRR={metrics.get('mrr', 0.0):.4f})"
                )

            def on_training_end(self, stats):
                print(f"[Debug] Training ended: {stats.get('stop_reason', 'Unknown')}")

        # Run 1: Baseline (No PC, No Logic)
        print("\n[Golden Fixture] Running Baseline (PC=0, Logic=0)...")
        config_base = DSLFMKGCConfig(
            num_entities=num_entities,
            num_relations=num_relations,
            entity_dim=128,
            feature_dim=128,
            max_communities=64,
            hidden_dim=256,
            lambda_logic=0.0,
            lambda_pc=0.0,
            contrastive_temperature=0.1,
            num_global_negatives=20,
            feature_weight=0.0,
            community_weight=1.0,
        )
        manager_base = DSLFMKGCManager(
            config_base, train_config, persistence_port=MockPersistence()
        )
        stats_base = manager_base.train(train_triples, valid_triples)
        mrr_base = stats_base.get("best_val_mrr", 0.0)
        print(f"[Golden Fixture] Baseline Result: MRR={mrr_base:.4f}")

        # Run 2: With PC (PC=0.1, Logic=0.1)
        print("\n[Golden Fixture] Running With PC (PC=0.1, Logic=0.1)...")
        config_pc = DSLFMKGCConfig(
            num_entities=num_entities,
            num_relations=num_relations,
            entity_dim=128,
            feature_dim=128,
            max_communities=64,
            hidden_dim=256,
            lambda_logic=0.1,
            lambda_pc=0.1,
            contrastive_temperature=0.1,
            num_global_negatives=20,
            feature_weight=0.0,
            community_weight=1.0,
        )
        manager_pc = DSLFMKGCManager(config_pc, train_config, persistence_port=MockPersistence())
        stats_pc = manager_pc.train(train_triples, valid_triples)
        mrr_pc = stats_pc.get("best_val_mrr", 0.0)
        print(f"[Golden Fixture] PC Result: MRR={mrr_pc:.4f}")

        # Assertions
        assert mrr_base > 0.15, f"Baseline DSLFM failed! MRR={mrr_base}"
        assert mrr_pc > 0.15, f"PC DSLFM failed! MRR={mrr_pc}"


if __name__ == "__main__":
    t = TestDSLFMHighScoreRepro()
    t.test_dslfm_learns_transitive_pattern_high_score()
