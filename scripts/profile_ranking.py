import cProfile
import io
import pstats

import numpy as np


class MockHandler:
    def __init__(self, n_relations: int, n_entities: int):
        self.n_relations = n_relations
        self.n_entities = n_entities

    def get_ranking(self, test_chunk, as_string=False):
        # Simulation of ranking results
        # Each query gets 10 candidates
        n_queries = len(test_chunk)
        results = {}
        for i in range(n_queries):
            query = tuple(test_chunk[i])
            results[query] = {
                "head": [(str(j), np.random.rand()) for j in range(10)],
                "tail": [(str(j), np.random.rand()) for j in range(10)],
            }
        return results


class Worker:
    def _collect_detailed_scores(self, handler, test_chunk):
        detailed_scores = []
        true_triples = set()
        for t in test_chunk:
            true_triples.add(("head", int(t[1]), int(t[2]), int(t[0])))
            true_triples.add(("tail", int(t[1]), int(t[0]), int(t[2])))

        all_rankings = handler.get_ranking(test_chunk, as_string=False)

        for triple_tuple, rankings in all_rankings.items():
            for direction in ["head", "tail"]:
                if direction in rankings:
                    source_id = str(triple_tuple[2] if direction == "head" else triple_tuple[0])
                    relation_id = str(triple_tuple[1])

                    for candidate_id, score in rankings[direction]:
                        r_int = int(relation_id)
                        s_int = int(source_id)
                        c_int = int(candidate_id)

                        is_true = 1 if (direction, r_int, s_int, c_int) in true_triples else 0

                        detailed_scores.append(
                            {
                                "direction": direction,
                                "rel_id": relation_id,
                                "src_id": source_id,
                                "cand_id": candidate_id,
                                "score": float(score),
                                "is_true": is_true,
                            }
                        )
        return detailed_scores


def profile_ranking_collection():
    print("Profiling Ranking Collection...")
    n_relations = 10
    n_entities = 10000

    test_chunk = np.random.randint(0, n_entities, (5000, 3))
    test_chunk[:, 1] = np.random.randint(0, n_relations, 5000)

    handler = MockHandler(n_relations, n_entities)
    worker = Worker()

    print("Starting profiling of _collect_detailed_scores...")

    pr = cProfile.Profile()
    pr.enable()

    worker._collect_detailed_scores(handler, test_chunk)

    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(20)
    print(s.getvalue())


if __name__ == "__main__":
    profile_ranking_collection()
