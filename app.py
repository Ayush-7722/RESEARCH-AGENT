import time

import streamlit as st

from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain

st.set_page_config(page_title="Multi-Agent Research System", page_icon="🔎", layout="wide")


def to_text(content) -> str:
    """Normalise LLM output (str, message object, or list of content blocks) to plain text."""
    if content is None:
        return ""
    if hasattr(content, "content"):
        content = content.content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return "\n".join(p for p in parts if p)
    return str(content)


def run_pipeline_with_ui(topic: str) -> dict:
    """Same 4 steps as pipeline.py, but reports progress to the Streamlit UI."""
    state = {}
    timings = {}

    # Step 1: search agent
    with st.status("Step 1 of 4: Search agent is searching the web...", expanded=False) as status:
        t0 = time.time()
        search_agent = build_search_agent()
        result = search_agent.invoke({
            "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
        })
        state["search_results"] = to_text(result["messages"][-1].content)
        timings["Search"] = time.time() - t0
        status.update(label=f"Step 1 of 4: Search done ({timings['Search']:.1f}s)", state="complete")

    # Step 2: reader agent
    with st.status("Step 2 of 4: Reader agent is scraping the best source...", expanded=False) as status:
        t0 = time.time()
        reader_agent = build_reader_agent()
        result = reader_agent.invoke({
            "messages": [("user",
                f"Based on the following search results about '{topic}', "
                f"pick the most relevant URL and scrape it for deeper content.\n\n"
                f"Search Results:\n{state['search_results'][:800]}"
            )]
        })
        state["scraped_content"] = to_text(result["messages"][-1].content)
        timings["Read"] = time.time() - t0
        status.update(label=f"Step 2 of 4: Reading done ({timings['Read']:.1f}s)", state="complete")

    # Step 3: writer chain
    with st.status("Step 3 of 4: Writer is drafting the report...", expanded=False) as status:
        t0 = time.time()
        research_combined = (
            f"SEARCH RESULTS : \n {state['search_results']} \n\n"
            f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
        )
        state["report"] = to_text(writer_chain.invoke({
            "topic": topic,
            "research": research_combined,
        }))
        timings["Write"] = time.time() - t0
        status.update(label=f"Step 3 of 4: Report written ({timings['Write']:.1f}s)", state="complete")

    # Step 4: critic chain
    with st.status("Step 4 of 4: Critic is reviewing the report...", expanded=False) as status:
        t0 = time.time()
        state["feedback"] = to_text(critic_chain.invoke({"report": state["report"]}))
        timings["Critique"] = time.time() - t0
        status.update(label=f"Step 4 of 4: Review done ({timings['Critique']:.1f}s)", state="complete")

    state["timings"] = timings
    state["topic"] = topic
    return state


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("How it works")
    st.markdown(
        "1. **Search agent** finds recent sources\n"
        "2. **Reader agent** scrapes the best one\n"
        "3. **Writer** drafts the report\n"
        "4. **Critic** reviews the report"
    )
    st.divider()
    if st.button("Clear results", use_container_width=True):
        st.session_state.pop("result", None)
        st.rerun()

# ---------------- Main ----------------
st.title("🔎 Multi-Agent Research System")
st.caption("Enter a topic and four agents will search, read, write and critique a report.")

with st.form("research_form"):
    topic = st.text_input(
        "Research topic",
        placeholder="e.g. Latest advances in solid-state batteries",
    )
    submitted = st.form_submit_button("Run research", type="primary")

if submitted:
    if not topic.strip():
        st.error("Please enter a topic first.")
    else:
        st.session_state.pop("result", None)
        try:
            st.session_state["result"] = run_pipeline_with_ui(topic.strip())
        except Exception as e:
            st.error(f"The pipeline failed: {e}")

result = st.session_state.get("result")

if result:
    total = sum(result["timings"].values())
    st.success(f"Done in {total:.1f}s for: {result['topic']}")

    tab_report, tab_feedback, tab_search, tab_scraped = st.tabs(
        ["Final report", "Critic feedback", "Search results", "Scraped content"]
    )

    with tab_report:
        st.markdown(result["report"])
        st.download_button(
            "Download report (.md)",
            data=result["report"],
            file_name="research_report.md",
            mime="text/markdown",
        )

    with tab_feedback:
        st.markdown(result["feedback"])

    with tab_search:
        st.markdown(result["search_results"])

    with tab_scraped:
        st.markdown(result["scraped_content"])

    with st.expander("Step timings"):
        for name, secs in result["timings"].items():
            st.write(f"{name}: {secs:.1f}s")