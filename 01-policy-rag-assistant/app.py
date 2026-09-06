"""
Demo UI.

    streamlit run app.py

Design intent: show the retrieval, don't hide it. A chat box that emits answers
from nowhere is unauditable. Every answer here can be traced to the passages
that produced it, and the retrieval mode can be switched live -- which makes
the hybrid-vs-vector difference something you can demonstrate rather than
merely assert.
"""

import streamlit as st

from rag import PolicyRAG
from llm_setup import describe

st.set_page_config(page_title="Policy Knowledge Assistant", layout="wide")


@st.cache_resource(show_spinner="Loading index and model...")
def load_engine(mode: str, k: int):
    return PolicyRAG(mode=mode, k=k)


st.title("Internal Policy Knowledge Assistant")
st.caption(
    "Grounded question answering over internal HR, finance, network and "
    "customer-care documents. Answers are generated only from retrieved "
    "passages and are traceable to their source."
)

with st.sidebar:
    st.subheader("Retrieval settings")
    mode = st.radio(
        "Mode",
        ["hybrid", "vector", "bm25"],
        help=(
            "vector = semantic only. bm25 = keyword only. "
            "hybrid = both, fused with Reciprocal Rank Fusion. "
            "Try asking about alarm code NE-5502 in each mode."
        ),
    )
    k = st.slider("Passages retrieved (k)", 1, 10, 4)
    st.divider()
    st.caption(describe())
    st.divider()
    st.subheader("Try asking")
    for q in [
        "How many days of annual leave do I get?",
        "What does alarm code NE-5502 mean?",
        "A fault affects 3000 subscribers. What severity and who do I escalate to?",
        "amar package change korte koto taka lagbe?",
        "What is the work from home policy?",
    ]:
        st.markdown(f"- {q}")

engine = load_engine(mode, k)

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("chunks"):
            with st.expander(f"Sources ({len(turn['chunks'])} passages retrieved)"):
                for i, c in enumerate(turn["chunks"], start=1):
                    st.markdown(
                        f"**[{i}] {c['source']}** — page {c['page']} "
                        f"· dept: `{c['department']}`"
                    )
                    st.text(c["text"][:600])
                    st.divider()

if question := st.chat_input("Ask about leave, expenses, network faults or tariffs..."):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            result = engine.ask(question)
        st.markdown(result.answer)

        chunks = [
            {
                "source": c.metadata.get("source", "?"),
                "page": c.metadata.get("page", 0),
                "department": c.metadata.get("department", "?"),
                "text": c.page_content,
            }
            for c in result.chunks
        ]
        with st.expander(f"Sources ({len(chunks)} passages retrieved)"):
            for i, c in enumerate(chunks, start=1):
                st.markdown(
                    f"**[{i}] {c['source']}** — page {c['page']} "
                    f"· dept: `{c['department']}`"
                )
                st.text(c["text"][:600])
                st.divider()

    st.session_state.history.append(
        {"role": "assistant", "content": result.answer, "chunks": chunks}
    )
