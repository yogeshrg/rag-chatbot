"""API routes for the RAG chatbot."""

import logging
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.models.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    DocumentInfo,
    DocumentUploadResponse,
    HealthResponse,
)
from app.services.rag_service import rag_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix=settings.api_prefix)


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    ollama_connected = await rag_service.check_ollama_connection()
    vector_store_ready = rag_service.is_ready()

    doc_count = 0
    if vector_store_ready:
        docs = await rag_service.list_documents()
        doc_count = len(docs)

    return HealthResponse(
        status="healthy" if ollama_connected and vector_store_ready else "degraded",
        version=settings.app_version,
        ollama_connected=ollama_connected,
        vector_store_ready=vector_store_ready,
        document_count=doc_count,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Non-streaming chat endpoint."""
    response = await rag_service.query(
        question=request.message,
        top_k=request.top_k,
        stream=False,
    )

    # Extract citations from source nodes
    citations = []
    if hasattr(response, "source_nodes"):
        for node in response.source_nodes:
            citations.append(f"{node.metadata.get('filename', 'unknown')}: {node.text[:200]}...")

    session_id = request.session_id or UUID(int=0)
    message = ChatMessage(
        role="assistant",
        content=str(response),
        citations=citations,
    )

    return ChatResponse(
        message=message,
        session_id=session_id,
        metadata={"model": settings.ollama_llm_model},
    )


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Streaming chat endpoint."""
    response = await rag_service.query(
        question=request.message,
        top_k=request.top_k,
        stream=True,
    )

    async def generate():
        async for token in response.async_response_gen():
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> DocumentUploadResponse:
    """Upload and index a document."""
    # Validate file extension
    file_ext = Path(file.filename or "").suffix.lower()
    if file_ext not in settings.supported_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Supported: {settings.supported_extensions}",
        )

    # Save file
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    file_path = settings.data_dir / file.filename
    content = await file.read()
    file_path.write_bytes(content)

    # Process in background
    result = await rag_service.add_documents([file_path])

    return DocumentUploadResponse(
        document_id=UUID(result["document_ids"][0]) if result["document_ids"] else UUID(int=0),
        filename=file.filename or "unknown",
        chunks_created=result["chunks"],
        status="indexed",
    )


@router.post("/documents/upload/batch", response_model=list[DocumentUploadResponse])
async def upload_documents_batch(
    files: list[UploadFile] = File(...),
) -> list[DocumentUploadResponse]:
    """Upload multiple documents."""
    results = []
    file_paths = []

    settings.data_dir.mkdir(parents=True, exist_ok=True)

    for file in files:
        file_ext = Path(file.filename or "").suffix.lower()
        if file_ext not in settings.supported_extensions:
            continue

        file_path = settings.data_dir / file.filename
        content = await file.read()
        file_path.write_bytes(content)
        file_paths.append(file_path)

    if file_paths:
        result = await rag_service.add_documents(file_paths)
        for i, doc_id in enumerate(result["document_ids"]):
            results.append(
                DocumentUploadResponse(
                    document_id=UUID(doc_id),
                    filename=files[i].filename or "unknown",
                    chunks_created=result["chunks"] // len(file_paths),
                    status="indexed",
                )
            )

    return results


@router.get("/documents", response_model=list[DocumentInfo])
async def list_documents() -> list[DocumentInfo]:
    """List all indexed documents."""
    docs = await rag_service.list_documents()
    return [
        DocumentInfo(
            document_id=UUID(doc["document_id"]),
            filename=doc["filename"],
            file_size=0,  # Would need to track separately
            chunks=doc["chunks"],
            created_at=None,  # Would need to track separately
            status="indexed",
        )
        for doc in docs
    ]


@router.delete("/documents/{document_id}")
async def delete_document(document_id: UUID) -> dict[str, str]:
    """Delete a document from the index."""
    success = await rag_service.delete_document(document_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found or deletion failed")
    return {"status": "deleted", "document_id": str(document_id)}


@router.post("/documents/reindex")
async def reindex_all() -> dict[str, str]:
    """Rebuild the entire index from source documents."""
    # This would re-process all files in data_dir
    file_paths = list(settings.data_dir.glob("*"))
    file_paths = [f for f in file_paths if f.suffix.lower() in settings.supported_extensions]

    if not file_paths:
        return {"status": "no_documents", "message": "No documents to index"}

    result = await rag_service.add_documents(file_paths)
    return {"status": "reindexed", "documents": str(result["added"])}
