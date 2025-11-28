import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pff.db.repositories.kg_rules import KGRulesRepository

@pytest.mark.asyncio
class TestKGRulesRepository:

    @pytest.fixture
    def mock_pool(self):
        class _AsyncContextManager:
            def __init__(self, result):
                self._result = result

            async def __aenter__(self):
                return self._result

            async def __aexit__(self, exc_type, exc, tb):
                return False

        pool = MagicMock()
        connection = AsyncMock()
        pool.acquire.return_value = _AsyncContextManager(connection)
        connection.transaction = MagicMock(return_value=_AsyncContextManager(None))
        return pool, connection

    async def test_ensure_schema(self, mock_pool):
        pool, conn = mock_pool
        
        with patch("pff.db.repositories.kg_rules.get_connection_pool", return_value=pool):
            repo = KGRulesRepository()
            repo.pool = pool # Inject mock pool directly
            
            await repo._ensure_schema()
            
            assert conn.execute.call_count >= 1
            assert "CREATE TABLE IF NOT EXISTS kg_rules" in conn.execute.call_args_list[0][0][0]

    async def test_save_rules(self, mock_pool):
        pool, conn = mock_pool
        
        with patch("pff.db.repositories.kg_rules.get_connection_pool", return_value=pool):
            repo = KGRulesRepository()
            repo.pool = pool
            repo._schema_ready = True # Skip schema check
            
            rules = [
                {"rule": "p(X,Y) :- q(X,Y)", "confidence": 0.9, "support": 10, "num_predictions": 100},
                {"rule": "a(X,Y) :- b(X,Z)", "confidence": 0.8, "support": 5, "num_predictions": 50}
            ]
            
            conn.copy_records_to_table.return_value = None
            
            count = await repo.save_rules(rules, source="test")
            
            assert count == 2
            conn.copy_records_to_table.assert_called_once()
            args = conn.copy_records_to_table.call_args
            assert args[0][0] == "kg_rules"
            assert len(args[1]['records']) == 2

    async def test_load_rules(self, mock_pool):
        pool, conn = mock_pool
        
        with patch("pff.db.repositories.kg_rules.get_connection_pool", return_value=pool):
            repo = KGRulesRepository()
            repo.pool = pool
            repo._schema_ready = True
            
            # Mock fetch return
            conn.fetch.return_value = [
                {"rule_text": "p(X,Y) :- q(X,Y)", "confidence": 0.9, "support": 10, "num_predictions": 100, "source": "test"},
                {"rule_text": "a(X,Y) :- b(X,Z)", "confidence": 0.8, "support": 5, "num_predictions": 50, "source": "test"}
            ]
            
            rules = await repo.load_rules(source="test")
            
            assert len(rules) == 2
            assert rules[0]["rule"] == "p(X,Y) :- q(X,Y)"
            assert rules[0]["confidence"] == 0.9
            
            conn.fetch.assert_called_once()
            assert "SELECT" in conn.fetch.call_args[0][0]

    async def test_delete_rules(self, mock_pool):
        pool, conn = mock_pool
        
        with patch("pff.db.repositories.kg_rules.get_connection_pool", return_value=pool):
            repo = KGRulesRepository()
            repo.pool = pool
            repo._schema_ready = True
            
            conn.execute.return_value = "DELETE 10"
            
            count = await repo.delete_rules(source="test")
            
            # The mock returns a string "DELETE 10", but the code parses it. 
            # However, asyncpg execute returns status string.
            # Our code does: result = await conn.execute(...) -> return int(result.split()[-1])
            
            assert count == 10
            conn.execute.assert_called_once()
            assert "DELETE FROM kg_rules" in conn.execute.call_args[0][0]
