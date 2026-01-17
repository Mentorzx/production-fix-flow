import cProfile
import pstats
import numpy as np
import io


# Mock RankingHandler
class MockHandler:
    def __init__(self, n_relations, n_entities, n_candidates=50):
        self.ranking = {}
        # Simulate ranking structure: {rel_id: {src_id: [(cand_id, score), ...]}}
        # Create a large ranking dictionary
        print("Generating mock ranking data...")
        for r in range(n_relations):
            self.ranking[str(r)] = {}
            # Simulate 1000 sources per relation
            for s in range(1000):
                # 50 candidates per source
                cands = []
                for _ in range(n_candidates):
                    c = np.random.randint(0, n_entities)
                    score = np.random.rand()
                    cands.append((str(c), score))
                self.ranking[str(r)][str(s)] = cands

    def get_ranking(self, as_string, direction):
        return self.ranking


class MockLogger:
    def info(self, msg):
        pass


class Worker:
    def __init__(self):
        self.logger = MockLogger()

    def _collect_detailed_scores(self, handler, test_chunk):
        detailed_scores = []
        test_triples_debug = set()
        for triple in test_chunk:
            h, r, t = int(triple[0]), int(triple[1]), int(triple[2])
            test_triples_debug.add((h, r, t))
        self.logger.info(f"Chunk de teste contém {len(test_triples_debug)} triplas únicas")
        test_set = set()
        for triple in test_chunk:
            test_set.add((int(triple[0]), int(triple[1]), int(triple[2])))
        # Build a set of true triples for quick lookup
        true_triples = set()
        for triple in test_chunk:
            h, r, t = int(triple[0]), int(triple[1]), int(triple[2])
            true_triples.add(("head", r, t, h))
            true_triples.add(("tail", r, h, t))
        self.logger.info(f"Conjunto de triplas verdadeiras criado com {len(true_triples)} entradas")

        for direction in ["head", "tail"]:
            ranking = handler.get_ranking(as_string=False, direction=direction)
            for relation_id, source_dictionary in ranking.items():
                for source_id, candidate_scores in source_dictionary.items():
                    if not candidate_scores:
                        continue
                    for candidate_id, score in candidate_scores:
                        if direction == "tail":
                            triple = (
                                int(source_id),
                                int(relation_id),
                                int(candidate_id),
                            )
                        else:  # direction == "head"
                            triple = (
                                int(candidate_id),
                                int(relation_id),
                                int(source_id),
                            )
                        r_int = int(relation_id)
                        s_int = int(source_id)
                        c_int = int(candidate_id)

                        is_true = (
                            1 if (direction, r_int, s_int, c_int) in true_triples else 0
                        )


                            in true_triples
                            else 0
                        )

                        r_int = int(relation_id)
                        s_int = int(source_id)
                        c_int = int(candidate_id)

                        is_true = 1 if (direction, r_int, s_int, c_int) in true_triples else 0

                        # Fix logic to match what seems intended (checking if the candidate completes a true triple)
                        # In true_triples, we stored: ("head", r, t, h) and ("tail", r, h, t)
                        # where source is the query entity (t for head query, h for tail query? No.)
                        # Standard KGC:
                        # Tail prediction: (?, r, t) -> Source is h. Candidate is t.
                        # Head prediction: (h, r, ?) -> Source is t. Candidate is h.
                        # true_triples construction:
                        # true_triples.add(("head", r, t, h)) -> implies query (?, r, t) -> answer h. Source=t?
                        # true_triples.add(("tail", r, h, t)) -> implies query (h, r, ?) -> answer t. Source=h.

                        # So if direction="tail": source=h, candidate=t. key=("tail", r, h, t)
                        # if direction="head": source=t, candidate=h. key=("head", r, t, h)

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

    # Generate test chunk (queries)
    # 5000 test triples
    test_chunk = np.random.randint(0, n_entities, (5000, 3))
    # Ensure relations are valid
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
