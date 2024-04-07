"""Streamlit UI for RAG Chatbot."""

import asyncio
from uuid import uuid4

import httpx
import streamlit as st

# Page config
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants
API_BASE = "http://localhost:8000/api/v1"


def init_session_state():
    """Initialize session state."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "health" not in st.session_state:
        st.session_state.health = None


async def check_health() -> dict:
    """Check API health."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{API_BASE}/health")
            return response.json()
    except Exception as e:
        return {"status": "error", "detail": str(e)}


async def send_message(message: str, stream: bool = True) -> dict:
    """Send message to API."""
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            if stream:
                async with client.stream(
                    "POST",
                    f"{API_BASE}/chat/stream",
                    json={
                        "message": message,
                        "session_id": st.session_state.session_id,
                        "stream": True,
                    },
                ) as response:
                    full_response = ""
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            full_response += data
                    return {"response": full_response, "citations": []}
            else:
                response = await client.post(
                    f"{API_BASE}/chat",
                    json={
                        "message": message,
                        "session_id": st.session_state.session_id,
                        "stream": False,
                    },
                )
                return response.json()
    except Exception as e:
        return {"error": str(e)}


async def upload_files(files) -> list:
    """Upload files to API."""
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            files_data = []
            for f in files:
                files_data.append(("files", (f.name, f.getvalue(), f.type)))
            response = await client.post(f"{API_BASE}/documents/upload/batch", files=files_data)
            return response.json()
    except Exception as e:
        return [{"error": str(e)}]


async def list_documents() -> list:
    """List indexed documents."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{API_BASE}/documents")
            return response.json()
    except Exception:
        return []


def render_sidebar():
    """Render sidebar with health and document management."""
    with st.sidebar:
        st.header("🤖 RAG Chatbot")
        st.caption("Local LLM with document knowledge")

        # Health check
        if st.button("🔄 Check Health", use_container_width=True):
            health = asyncio.run(check_health())
            st.session_state.health = health

        if st.session_state.health:
            status = st.session_state.health.get("status", "unknown")
            if status == "healthy":
                st.success("✅ System Healthy")
            elif status == "degraded":
                st.warning("⚠️ Degraded")
            else:
                st.error("❌ Unhealthy")

            st.caption(
                f"Ollama: {'🟢' if st.session_state.health.get('ollama_connected') else '🔴'}"
            )
            st.caption(
                f"Vector Store: {'🟢' if st.session_state.health.get('vector_store_ready') else '🔴'}"
            )
            st.caption(f"Documents: {st.session_state.health.get('document_count', 0)}")

        st.divider()

        # Document upload
        st.subheader("📄 Documents")
        uploaded_files = st.file_uploader(
            "Upload documents",
            accept_multiple_files=True,
            type=["pdf", "txt", "md", "py", "js", "ts", "json", "yaml", "yml"],
        )
        if uploaded_files and st.button("📤 Upload & Index", use_container_width=True):
            with st.spinner("Uploading and indexing..."):
                results = asyncio.run(upload_files(uploaded_files))
                for r in results:
                    if "error" in r:
                        st.error(f"Error: {r['error']}")
                    else:
                        st.success(f"✅ {r['filename']}: {r['chunks_created']} chunks")

        st.divider()

        # Document list
        if st.button("📋 List Documents", use_container_width=True):
            docs = asyncio.run(list_documents())
            if docs:
                for doc in docs:
                    st.caption(f"📄 {doc['filename']} ({doc['chunks']} chunks)")
            else:
                st.caption("No documents indexed")

        st.divider()

        # Session controls
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid4())
            st.rerun()


def render_chat():
    """Render chat interface."""
    # Display messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("citations"):
                with st.expander("📚 Sources"):
                    for cite in msg["citations"]:
                        st.caption(cite)

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = asyncio.run(send_message(prompt, stream=False))

            if "error" in result:
                st.error(f"Error: {result['error']}")
                return

            response_text = result.get("response", "") or result.get("message", {}).get(
                "content", ""
            )
            citations = result.get("citations", []) or result.get("message", {}).get(
                "citations", []
            )

            st.markdown(response_text)

            if citations:
                with st.expander("📚 Sources"):
                    for cite in citations:
                        st.caption(cite)

        # Add assistant message
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response_text,
                "citations": citations,
            }
        )


def main():
    """Main Streamlit app."""
    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
