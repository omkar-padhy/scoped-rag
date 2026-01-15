"""Streamlit frontend for RAG system"""
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Scoped RAG", page_icon="📚")
st.title("📚 Scoped RAG")
st.caption("Ask questions about your documents")

# Sidebar
with st.sidebar:
    st.header("Settings")
    show_sources = st.checkbox("Show sources", value=True)
    show_context = st.checkbox("🔍 Show retrieved context (debug)", value=False)
    
    # Access Level
    st.subheader("🔐 Access Control")
    user_level = st.select_slider(
        "User Security Level",
        options=[1, 2, 3, 4, 5],
        value=2,
        help="L1: Public | L2: Internal | L3: Confidential | L4: Sensitive | L5: Top Secret"
    )
    
    # File Upload
    st.header("📁 Upload Files")
    uploaded_files = st.file_uploader(
        "Upload PDF or Images",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                r = requests.post(f"{API_URL}/upload", files=files)
                if r.ok:
                    st.success(f"✓ {uploaded_file.name}")
                else:
                    st.error(f"✗ {uploaded_file.name}: {r.text}")
            except:
                st.error(f"✗ {uploaded_file.name}: API error")
    
    # Show existing files
    st.header("📂 Files")
    try:
        r = requests.get(f"{API_URL}/files")
        if r.ok:
            files = r.json().get("files", [])
            if files:
                for f in files:
                    col1, col2 = st.columns([3, 1])
                    col1.write(f"📄 {f}")
                    if col2.button("🗑️", key=f"del_{f}"):
                        requests.delete(f"{API_URL}/files/{f}")
                        st.rerun()
            else:
                st.caption("No files yet")
    except:
        st.caption("API not running")
    
    st.divider()
    
    if st.button("🔄 Sync Index"):
        with st.spinner("Syncing..."):
            try:
                r = requests.post(f"{API_URL}/reindex") # Endpoint name stays same for compatibility
                if r.ok:
                    data = r.json()
                    st.success(f"Done! ({data.get('status')})")
                else:
                    st.error(r.text)
            except:
                st.error("API not running")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources"):
                for s in msg["sources"]:
                    st.write(f"- {s}")
        if msg.get("context"):
            with st.expander("🔍 Retrieved Context (Debug)"):
                chunks = msg["context"].split("\n\n---\n\n")
                for i, chunk in enumerate(chunks, 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.code(chunk[:500] + ("..." if len(chunk) > 500 else ""), language=None)

# Chat input
if question := st.chat_input("Ask a question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    
    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                endpoint = "/query-with-sources" if show_sources else "/query"
                payload = {
                    "question": question,
                    "user_level": user_level
                }
                r = requests.post(f"{API_URL}{endpoint}", json=payload)
                
                if r.ok:
                    data = r.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])
                    context = data.get("context", "")
                    
                    st.write(answer)
                    
                    if show_sources and sources:
                        with st.expander("📎 Sources"):
                            for s in sources:
                                st.write(f"- {s}")
                    
                    if show_context and context:
                        with st.expander("🔍 Retrieved Context (Debug)", expanded=False):
                            st.markdown("**Raw chunks sent to LLM:**")
                            # Split by separator and show each chunk
                            chunks = context.split("\n\n---\n\n")
                            for i, chunk in enumerate(chunks, 1):
                                st.markdown(f"**Chunk {i}:**")
                                st.code(chunk[:500] + ("..." if len(chunk) > 500 else ""), language=None)
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources if show_sources else None,
                        "context": context if show_context else None,
                    })
                else:
                    st.error(f"Error: {r.text}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Run: `python server.py`")

