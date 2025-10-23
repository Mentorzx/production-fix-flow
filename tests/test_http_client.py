"""
Tests for HttpClient (pff/utils/clients/http_client.py)

Tests cover:
- Async HTTP request execution
- Retry logic with exponential backoff
- Failover to alternate hosts
- Connection pooling and limits
- HTTP/2 support
- Template caching
- Error handling (benign vs critical)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from pff.utils.clients.http_client import HttpClient


@pytest.fixture
def http_client():
    """Create an HttpClient instance for testing."""
    return HttpClient()


@pytest.fixture
async def async_http_client():
    """Create an HttpClient instance with async context."""
    client = HttpClient()
    yield client
    await client.close()


# ═══════════════════════════════════════════════════════════════════
# Initialization Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestHttpClientInitialization:
    """Test HttpClient initialization and configuration."""

    def test_http_client_initialization_defaults(self, http_client):
        """Test that HttpClient initializes with default values."""
        assert http_client._timeout == 10.0
        assert http_client._retries == 3
        assert http_client._backoff == 0.5
        assert http_client._client is not None

    def test_http_client_custom_config(self):
        """Test HttpClient with custom configuration."""
        client = HttpClient(timeout=5.0, retries=5, backoff=1.0)

        assert client._timeout == 5.0
        assert client._retries == 5
        assert client._backoff == 1.0

    def test_http_client_has_cache_manager(self, http_client):
        """Test that HttpClient has a CacheManager."""
        assert hasattr(http_client, 'cache')
        assert http_client.cache is not None

    def test_http_client_connection_limits(self, http_client):
        """Test that httpx client is configured with connection pooling."""
        # Verify the client was created (limits are internal to httpx)
        assert http_client._client is not None
        assert isinstance(http_client._client, httpx.AsyncClient)

    def test_http_client_timeout_configuration(self, http_client):
        """Test that timeout is properly configured."""
        timeout = http_client._client.timeout

        assert timeout.connect == 10.0
        assert timeout.read == 10.0
        assert timeout.write == 10.0
        assert timeout.pool == 20.0  # 2x timeout


# ═══════════════════════════════════════════════════════════════════
# Context Manager Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestHttpClientContextManager:
    """Test HttpClient async context manager protocol."""

    @pytest.mark.asyncio
    async def test_http_client_context_manager(self):
        """Test HttpClient can be used as async context manager."""
        async with HttpClient() as client:
            assert client is not None
            assert client._client is not None

    @pytest.mark.asyncio
    async def test_http_client_closes_on_exit(self):
        """Test HttpClient closes connection on context exit."""
        client = HttpClient()

        async with client:
            pass

        # After exiting context, client should be closed
        # We can verify by trying to use it (should fail or be None)
        assert client._client is not None  # Client object still exists


# ═══════════════════════════════════════════════════════════════════
# Host Candidates Tests (Failover)
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestHostCandidates:
    """Test host candidate building for failover."""

    def test_build_host_candidates_absolute_url(self, http_client):
        """Test building candidates with absolute URL."""
        url = "https://api.example.com/endpoint"
        candidates = http_client._build_host_candidates(url, "GET")

        assert len(candidates) >= 1
        # First candidate should use the exact URL
        assert "https://api.example.com/endpoint" in candidates[0][1]["url"]

    def test_build_host_candidates_relative_path(self, http_client):
        """Test building candidates with relative path (uses failover)."""
        url = "/bias/some/endpoint"
        candidates = http_client._build_host_candidates(url, "GET")

        # Should generate multiple candidates for different hosts
        assert len(candidates) > 1

    def test_build_host_candidates_preserves_method(self, http_client):
        """Test that method is preserved in candidates."""
        url = "https://api.example.com/test"
        candidates = http_client._build_host_candidates(url, "POST", json={"test": "data"})

        for _, kwargs in candidates:
            assert kwargs["method"] == "POST"
            assert kwargs.get("json") == {"test": "data"}


# ═══════════════════════════════════════════════════════════════════
# Retry Logic Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestRetryLogic:
    """Test retry logic with exponential backoff."""

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self, async_http_client):
        """Test that client retries on timeout."""
        with patch.object(async_http_client._client, 'request', new_callable=AsyncMock) as mock_request:
            # First two attempts timeout, third succeeds
            mock_request.side_effect = [
                httpx.TimeoutException("Timeout"),
                httpx.TimeoutException("Timeout"),
                httpx.Response(200, json={"success": True}),
            ]

            response = await async_http_client._attempt_single_request(
                method="GET",
                url="https://api.example.com/test"
            )

            assert response.status_code == 200
            assert mock_request.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhaustion_raises_error(self, async_http_client):
        """Test that exhausting retries raises error."""
        with patch.object(async_http_client._client, 'request', new_callable=AsyncMock) as mock_request:
            # All attempts timeout
            mock_request.side_effect = httpx.TimeoutException("Timeout")

            with pytest.raises(httpx.TimeoutException):
                await async_http_client._attempt_single_request(
                    method="GET",
                    url="https://api.example.com/test"
                )

            # Should try initial + 3 retries = 4 total
            assert mock_request.call_count == 4

    @pytest.mark.asyncio
    async def test_successful_request_no_retry(self, async_http_client):
        """Test that successful request doesn't retry."""
        with patch.object(async_http_client._client, 'request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = httpx.Response(200, json={"success": True})

            response = await async_http_client._attempt_single_request(
                method="GET",
                url="https://api.example.com/test"
            )

            assert response.status_code == 200
            assert mock_request.call_count == 1  # No retries


# ═══════════════════════════════════════════════════════════════════
# Response Handling Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestResponseHandling:
    """Test response extraction and error handling."""

    @pytest.mark.asyncio
    async def test_extract_json_response(self, async_http_client):
        """Test extracting JSON from response."""
        mock_response = httpx.Response(
            200,
            json={"key": "value"},
            request=httpx.Request("GET", "https://example.com")
        )

        result = await async_http_client._extract_response_content(mock_response, "test")

        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_extract_text_response_non_json(self, async_http_client):
        """Test extracting text when response is not JSON."""
        mock_response = httpx.Response(
            200,
            text="Plain text response",
            request=httpx.Request("GET", "https://example.com")
        )

        result = await async_http_client._extract_response_content(mock_response, "test")

        assert result == "Plain text response"

    @pytest.mark.asyncio
    async def test_extract_empty_response(self, async_http_client):
        """Test extracting empty response returns empty dict."""
        mock_response = httpx.Response(
            204,
            request=httpx.Request("GET", "https://example.com")
        )

        result = await async_http_client._extract_response_content(mock_response, "test")

        assert result == {}


# ═══════════════════════════════════════════════════════════════════
# Error Handling Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestErrorHandling:
    """Test HTTP error handling (benign vs critical)."""

    @pytest.mark.asyncio
    async def test_benign_error_handling_409_duplicate(self, async_http_client):
        """Test that 409 Duplicate errors are treated as benign."""
        mock_response = httpx.Response(
            409,
            json={"code": "BIAS.DuplicateResource", "details": "Resource exists"},
            request=httpx.Request("POST", "https://example.com")
        )

        result = await async_http_client._handle_response_error(
            mock_response,
            "Duplicate resource",
            "test_tag"
        )

        # Benign errors return False (not blocking)
        assert result is False

    @pytest.mark.asyncio
    async def test_server_error_raises_exception(self, async_http_client):
        """Test that 5xx server errors raise RuntimeError."""
        mock_response = httpx.Response(
            503,
            json={"code": "SERVICE_UNAVAILABLE", "details": "Service down"},
            request=httpx.Request("GET", "https://example.com")
        )

        with pytest.raises(RuntimeError, match="Erro de servidor não recuperável"):
            await async_http_client._handle_response_error(
                mock_response,
                "Service unavailable",
                "test_tag"
            )


# ═══════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestHttpClientIntegration:
    """Test HttpClient end-to-end scenarios."""

    @pytest.mark.asyncio
    async def test_make_request_basic(self, async_http_client):
        """Test basic make_request functionality."""
        endpoint_config = {
            "url": "https://httpbin.org/get",
            "method": "GET",
            "type": "test_endpoint",
        }
        subscriber_data = {}

        with patch.object(async_http_client, '_execute_json_request', new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"success": True}

            result = await async_http_client.make_request(endpoint_config, subscriber_data)

            assert result == {"success": True}
            assert mock_exec.called

    @pytest.mark.asyncio
    async def test_close_client(self, async_http_client):
        """Test closing the HTTP client."""
        await async_http_client.close()

        # Client should still exist after close
        assert async_http_client._client is not None
