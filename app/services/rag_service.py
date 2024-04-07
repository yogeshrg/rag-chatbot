"""RAG pipeline service using LlamaIndex and Ollama."""

import logging
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import faiss
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.faiss import FaissVectorStore

from app.core.config import settings

logger = logging.getLogger(__name__)


class RAGService:
    """Service for managing the RAG pipeline."""

    def __init__(self) -> None:
        self._index: VectorStoreIndex | None = None
        self._query_engine: RetrieverQueryEngine | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the RAG pipeline."""
        if self._initialized:
            return

        logger.info("Initializing RAG pipeline...")

        # Configure LlamaIndex settings
        Settings.llm = Ollama(
            model=settings.ollama_llm_model,
            base_url=settings.ollama_base_url,
            request_timeout=settings.ollama_timeout,
            temperature=0.7,
        )
        Settings.embed_model = OllamaEmbedding(
            model_name=settings.ollama_embed_model,
            base_url=settings.ollama_base_url,
            request_timeout=settings.ollama_timeout,
        )
        Settings.node_parser = SentenceSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        # Ensure vector store directory exists
        settings.vector_store_path.mkdir(parents=True, exist_ok=True)

        # Load or create vector store
        await self._load_or_create_index()

        # Create query engine
        retriever = VectorIndexRetriever(
            index=self._index,
            similarity_top_k=settings.top_k,
        )
        self._query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=settings.similarity_threshold)
            ],
            streaming=settings.stream_response,
        )

        self._initialized = True
        logger.info("RAG pipeline initialized successfully")

    async def _load_or_create_index(self) -> None:
        """Load existing index or create new one."""
        index_file = settings.vector_store_path / f"{settings.index_name}.faiss"
        docstore_file = settings.vector_store_path / f"{settings.index_name}_docstore.json"

        if index_file.exists() and docstore_file.exists():
            logger.info("Loading existing vector index...")
            vector_store = FaissVectorStore.from_persist_dir(str(settings.vector_store_path))
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                persist_dir=str(settings.vector_store_path),
            )
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context,
            )
        else:
            logger.info("Creating new vector index...")
            # Create empty FAISS index
            embed_dim = 768  # nomic-embed-text dimension
            faiss_index = faiss.IndexFlatIP(embed_dim)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            # Explicitly pass embed_model to avoid OpenAI default
            self._index = VectorStoreIndex(
                [],
                storage_context=storage_context,
                embed_model=Settings.embed_model,
            )

        # Persist the empty index structure
        self._index.storage_context.persist(persist_dir=str(settings.vector_store_path))

    async def add_documents(self, file_paths: list[Path]) -> dict[str, Any]:
        """Add documents to the index."""
        if not self._initialized:
            await self.initialize()

        logger.info(f"Adding {len(file_paths)} documents to index...")
        documents = []

        for file_path in file_paths:
            if not file_path.exists():
                logger.warning(f"File not found: {file_path}")
                continue

            # Load document
            reader = SimpleDirectoryReader(input_files=[str(file_path)])
            docs = reader.load_data()
            for doc in docs:
                doc.metadata["source"] = str(file_path)
                doc.metadata["filename"] = file_path.name
            documents.extend(docs)

        if not documents:
            return {"added": 0, "chunks": 0, "document_ids": []}

        # Insert into index
        document_ids = []
        for doc in documents:
            doc_id = uuid4()
            doc.metadata["document_id"] = str(doc_id)
            document_ids.append(doc_id)

        self._index.insert_nodes(documents)
        self._index.storage_context.persist(persist_dir=str(settings.vector_store_path))

        logger.info(f"Added {len(documents)} document chunks")
        return {
            "added": len(documents),
            "chunks": len(documents),
            "document_ids": [str(d) for d in document_ids],
        }

    async def query(
        self,
        question: str,
        top_k: int | None = None,
        stream: bool = True,
    ) -> Any:
        """Query the RAG system."""
        if not self._initialized:
            await self.initialize()

        # Override top_k if provided
        if top_k is not None:
            retriever = VectorIndexRetriever(
                index=self._index,
                similarity_top_k=top_k,
            )
            query_engine = RetrieverQueryEngine.from_args(
                retriever=retriever,
                node_postprocessors=[
                    SimilarityPostprocessor(similarity_cutoff=settings.similarity_threshold)
                ],
                streaming=stream,
            )
        else:
            query_engine = self._query_engine

        logger.info(f"Querying: {question[:100]}...")
        response = await query_engine.aquery(question)
        return response

    async def list_documents(self) -> list[dict[str, Any]]:
        """List all indexed documents."""
        if not self._initialized:
            await self.initialize()

        # Extract unique documents from index
        doc_map = {}
        if self._index and self._index.docstore:
            for node in self._index.docstore.docs.values():
                doc_id = node.metadata.get("document_id")
                filename = node.metadata.get("filename", "unknown")
                if doc_id and doc_id not in doc_map:
                    doc_map[doc_id] = {
                        "document_id": doc_id,
                        "filename": filename,
                        "chunks": 0,
                    }
                if doc_id:
                    doc_map[doc_id]["chunks"] += 1

        return list(doc_map.values())

    async def delete_document(self, document_id: UUID) -> bool:
        """Delete a document from the index."""
        if not self._initialized:
            await self.initialize()

        # LlamaIndex doesn't support direct deletion easily
        # Would need to rebuild index without the document
        logger.warning("Document deletion not fully implemented - requires index rebuild")
        return False

    def is_ready(self) -> bool:
        """Check if the service is ready."""
        return self._initialized and self._index is not None

    async def check_ollama_connection(self) -> bool:
        """Check if Ollama is accessible."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{settings.ollama_base_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False


# Global service instance
rag_service = RAGService()
