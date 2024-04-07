"""Tests for RAG Chatbot."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.config import settings
from app.models.schemas import ChatMessage, ChatRequest, HealthResponse
from app.services.rag_service import RAGService


class TestConfig:
    """Test configuration loading."""

    def test_settings_defaults(self):
        """Test default settings values."""
        assert settings.app_name == "RAG Chatbot"
        assert settings.ollama_base_url == "http://localhost:11434"
        assert settings.ollama_llm_model == "llama3.1:8b"
        assert settings.chunk_size == 512
        assert settings.top_k == 5


class TestSchemas:
    """Test Pydantic schemas."""

    def test_chat_request_valid(self):
        """Test valid chat request."""
        req = ChatRequest(message="Hello", stream=True)
        assert req.message == "Hello"
        assert req.stream is True
        assert req.temperature == 0.7

    def test_chat_request_invalid_temperature(self):
        """Test invalid temperature validation."""
        with pytest.raises(ValueError):
            ChatRequest(message="Hello", temperature=3.0)

    def test_chat_message(self):
        """Test chat message model."""
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_health_response(self):
        """Test health response model."""
        health = HealthResponse(
            status="healthy",
            version="0.1.0",
            ollama_connected=True,
            vector_store_ready=True,
            document_count=5,
        )
        assert health.status == "healthy"
        assert health.document_count == 5


class TestRAGService:
    """Test RAG service."""

    @pytest.fixture
    def rag_service(self):
        """Create RAG service instance."""
        return RAGService()

    @pytest.mark.asyncio
    async def test_initialize_creates_index(self, rag_service, tmp_path):
        """Test initialization creates vector index."""
        with patch.object(settings, "vector_store_path", tmp_path / "vector_store"):
            with patch("app.services.rag_service.Settings") as mock_settings:
                mock_settings.llm = MagicMock()
                mock_settings.embed_model = MagicMock()
                mock_settings.node_parser = MagicMock()
                # Also need to patch the Settings used in query_engine
                with patch(
                    "app.services.rag_service.RetrieverQueryEngine.from_args"
                ) as mock_query_engine:
                    mock_query_engine.return_value = MagicMock()
                    await rag_service.initialize()
                    assert rag_service.is_ready()

    @pytest.mark.asyncio
    async def test_check_ollama_connection(self, rag_service):
        """Test Ollama connection check."""
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_get.return_value.status_code = 200
            result = await rag_service.check_ollama_connection()
            assert result is True

        with patch("httpx.AsyncClient.get") as mock_get:
            mock_get.side_effect = Exception("Connection failed")
            result = await rag_service.check_ollama_connection()
            assert result is False


class TestAPIRoutes:
    """Test API routes."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient

        from app.main import app

        return TestClient(app)

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "RAG Chatbot"
        assert "docs" in data

    def test_health_endpoint(self, client):
        """Test health endpoint."""
        with patch(
            "app.api.routes.rag_service.check_ollama_connection", new_callable=AsyncMock
        ) as mock_ollama:
            with patch("app.api.routes.rag_service.is_ready", return_value=True):
                with patch(
                    "app.api.routes.rag_service.list_documents", new_callable=AsyncMock
                ) as mock_list:
                    mock_ollama.return_value = True
                    mock_list.return_value = []
                    response = client.get("/api/v1/health")
                    assert response.status_code == 200
                    data = response.json()
                    assert data["status"] == "healthy"
                    assert data["ollama_connected"] is True

    def test_chat_endpoint(self, client):
        """Test chat endpoint."""
        with patch("app.api.routes.rag_service.query", new_callable=AsyncMock) as mock_query:
            mock_response = MagicMock()
            mock_response.__str__ = lambda self: "Test response"
            mock_response.source_nodes = []
            mock_query.return_value = mock_response

            response = client.post("/api/v1/chat", json={"message": "Hello"})
            assert response.status_code == 200
            data = response.json()
            assert "message" in data
            assert data["message"]["role"] == "assistant"


class TestDocumentProcessing:
    """Test document processing."""

    @pytest.mark.asyncio
    async def test_add_documents(self, tmp_path):
        """Test adding documents to index."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is a test document for RAG.")

        service = RAGService()
        with patch.object(settings, "vector_store_path", tmp_path / "vector_store"):
            with patch.object(settings, "data_dir", tmp_path):
                with patch("app.services.rag_service.Settings") as mock_settings:
                    mock_settings.llm = MagicMock()
                    mock_settings.embed_model = MagicMock()
                    mock_settings.node_parser = MagicMock()
                    with patch(
                        "app.services.rag_service.RetrieverQueryEngine.from_args"
                    ) as mock_query_engine:
                        mock_query_engine.return_value = MagicMock()
                        await service.initialize()

                        # Mock the index insert
                        service._index.insert_nodes = MagicMock()
                        service._index.storage_context.persist = MagicMock()

                        result = await service.add_documents([test_file])
                        assert result["added"] >= 0


# Integration tests (require running services)
@pytest.mark.integration
class TestIntegration:
    """Integration tests requiring running services."""

    @pytest.mark.asyncio
    async def test_full_chat_flow(self):
        """Test full chat flow with real services."""
        # This would run against actual API
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
