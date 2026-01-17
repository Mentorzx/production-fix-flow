"""Tests for pff/shared/research.py - TripleStore and Research utilities."""

from pff.shared.research import TripleStore


class TestTripleStore:
    """Tests for TripleStore triple indexing."""

    def test_triplestore_empty_init(self):
        """Verify empty TripleStore can be created."""
        store = TripleStore()
        assert len(store) == 0

    def test_triplestore_init_with_triples(self):
        """Verify TripleStore can be initialized with triples."""
        triples = [("s1", "p1", "o1"), ("s2", "p2", "o2")]
        store = TripleStore(triples)
        assert len(store) == 2

    def test_triplestore_add_single_triple(self):
        """Verify single triple can be added."""
        store = TripleStore()
        store.add(("subject", "predicate", "object"))
        assert len(store) == 1

    def test_triplestore_add_duplicate_triple(self):
        """Verify duplicate triples are handled (not counted twice)."""
        store = TripleStore()
        store.add(("s", "p", "o"))
        store.add(("s", "p", "o"))
        assert len(store) == 1

    def test_triplestore_find_by_subject(self):
        """Verify find by subject works."""
        triples = [("s1", "p1", "o1"), ("s1", "p2", "o2"), ("s2", "p1", "o3")]
        store = TripleStore(triples)
        results = list(store.find(s="s1"))
        assert len(results) == 2
        assert all(t[0] == "s1" for t in results)

    def test_triplestore_find_by_predicate(self):
        """Verify find by predicate works."""
        triples = [("s1", "p1", "o1"), ("s2", "p1", "o2"), ("s3", "p2", "o3")]
        store = TripleStore(triples)
        results = list(store.find(p="p1"))
        assert len(results) == 2
        assert all(t[1] == "p1" for t in results)

    def test_triplestore_find_by_object(self):
        """Verify find by object works."""
        triples = [("s1", "p1", "o1"), ("s2", "p2", "o1"), ("s3", "p3", "o2")]
        store = TripleStore(triples)
        results = list(store.find(o="o1"))
        assert len(results) == 2
        assert all(t[2] == "o1" for t in results)

    def test_triplestore_find_by_subject_predicate(self):
        """Verify find by subject and predicate works."""
        triples = [("s1", "p1", "o1"), ("s1", "p1", "o2"), ("s1", "p2", "o3")]
        store = TripleStore(triples)
        results = list(store.find(s="s1", p="p1"))
        assert len(results) == 2
        assert all(t[0] == "s1" and t[1] == "p1" for t in results)

    def test_triplestore_find_by_subject_object(self):
        """Verify find by subject and object works."""
        triples = [("s1", "p1", "o1"), ("s1", "p2", "o1"), ("s2", "p1", "o1")]
        store = TripleStore(triples)
        results = list(store.find(s="s1", o="o1"))
        assert len(results) == 2
        assert all(t[0] == "s1" and t[2] == "o1" for t in results)

    def test_triplestore_find_by_predicate_object(self):
        """Verify find by predicate and object works."""
        triples = [("s1", "p1", "o1"), ("s2", "p1", "o1"), ("s3", "p1", "o2")]
        store = TripleStore(triples)
        results = list(store.find(p="p1", o="o1"))
        assert len(results) == 2
        assert all(t[1] == "p1" and t[2] == "o1" for t in results)

    def test_triplestore_find_exact_triple(self):
        """Verify exact triple lookup works."""
        triples = [("s1", "p1", "o1"), ("s2", "p2", "o2")]
        store = TripleStore(triples)
        results = list(store.find(s="s1", p="p1", o="o1"))
        assert len(results) == 1
        assert results[0] == ("s1", "p1", "o1")

    def test_triplestore_find_nonexistent_triple(self):
        """Verify nonexistent triple returns empty."""
        triples = [("s1", "p1", "o1")]
        store = TripleStore(triples)
        results = list(store.find(s="nonexistent"))
        assert len(results) == 0

    def test_triplestore_find_all(self):
        """Verify find with no params returns all triples."""
        triples = [("s1", "p1", "o1"), ("s2", "p2", "o2")]
        store = TripleStore(triples)
        results = list(store.find())
        assert len(results) == 2

    def test_triplestore_deduplicates_on_init(self):
        """Verify duplicate triples are removed on initialization."""
        triples = [("s", "p", "o"), ("s", "p", "o"), ("s", "p", "o")]
        store = TripleStore(triples)
        assert len(store) == 1

    def test_triplestore_multiple_objects_same_sp(self):
        """Verify multiple objects for same subject-predicate."""
        store = TripleStore()
        store.add(("entity", "has_name", "name1"))
        store.add(("entity", "has_name", "name2"))
        results = list(store.find(s="entity", p="has_name"))
        assert len(results) == 2
        objects = {t[2] for t in results}
        assert objects == {"name1", "name2"}

    def test_triplestore_kg_style_relations(self):
        """Verify KG-style relations work correctly."""
        triples = [
            ("customer_1", "has_contract", "contract_1"),
            ("customer_1", "has_service", "service_1"),
            ("contract_1", "has_status", "active"),
        ]
        store = TripleStore(triples)

        # Find all relations for customer_1
        customer_relations = list(store.find(s="customer_1"))
        assert len(customer_relations) == 2

        # Find all active statuses
        active = list(store.find(p="has_status", o="active"))
        assert len(active) == 1


class TestTripleStoreIndexes:
    """Tests for TripleStore index structures."""

    def test_triplestore_spo_index(self):
        """Verify SPO index is populated correctly."""
        store = TripleStore([("s", "p", "o")])
        assert "o" in store.spo["s"]["p"]

    def test_triplestore_pos_index(self):
        """Verify POS index is populated correctly."""
        store = TripleStore([("s", "p", "o")])
        assert "s" in store.pos["p"]["o"]

    def test_triplestore_osp_index(self):
        """Verify OSP index is populated correctly."""
        store = TripleStore([("s", "p", "o")])
        assert "p" in store.osp["o"]["s"]
