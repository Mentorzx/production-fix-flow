import torch
import time
from pff.domain.learning.dslfm.dslfm_kgc import DSLFMKGCConfig, DSLFMKGCModel
from pff.shared.system.cuda import is_cuda_available


def verify_model_export():
    device = torch.device("cuda" if is_cuda_available() else "cpu")
    print(f"Device: {device}")

    n_entities = 1000
    n_relations = 10

    config = DSLFMKGCConfig(
        num_entities=n_entities,
        num_relations=n_relations,
        num_triples=5000,
        entity_dim=64,
        feature_dim=64,
        hidden_dim=128,
        use_checkpointing=False,
        lambda_logic=0.0,
        lambda_pc=0.0,
    )

    model = DSLFMKGCModel(config).to(device)
    model.eval()

    # Create example input
    heads = torch.randint(0, n_entities, (32,), device=device)
    relations = torch.randint(0, n_relations, (32,), device=device)
    tails = torch.randint(0, n_entities, (32,), device=device)
    example_triples = torch.stack([heads, relations, tails], dim=1)

    print("Testing export_inference_model...")
    try:
        start_export = time.perf_counter()
        exported_program = model.export_inference_model(example_triples)
        print(f"Export successful. Time: {time.perf_counter() - start_export:.4f}s")

        # Verify execution
        print("Verifying execution...")
        with torch.no_grad():
            eager_out = model.score_triples_batch(example_triples)
            exported_out = exported_program.module()(example_triples)

            # Check correctness
            if torch.allclose(eager_out, exported_out, atol=1e-5):
                print("Output matches eager execution.")
            else:
                print("WARNING: Output mismatch!")
                print(f"Eager: {eager_out[:5]}")
                print(f"Export: {exported_out[:5]}")

            # Verify dynamic shapes (batch size 16)
            print("Verifying dynamic shapes...")
            batch16 = example_triples[:16]
            out16 = exported_program.module()(batch16)
            if out16.shape[0] == 16:
                print("Dynamic shape handling verified.")
            else:
                print(f"Dynamic shape failed. Expected 16, got {out16.shape[0]}")

    except Exception as e:
        print(f"Export method failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    verify_model_export()
