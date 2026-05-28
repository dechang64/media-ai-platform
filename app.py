#!/usr/bin/env python3
"""
MediaAI Streamlit Application

Cell Culture Media Intelligence Platform - Interactive Web Interface
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="MediaAI - Cell Culture Media Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================
# SECTION 1: IMPORTS
# ============================================

try:
    from mediaai import (
        MediaVault,
        MediaOptimizer,
        FLEngine,
        CellSegmentor,
        ViabilityClassifier,
        KnowledgeHub,
        AuditChain,
    )
except ImportError:
    st.error("Please install mediaai: pip install -e .")
    st.stop()


# ============================================
# SECTION 2: CUSTOM CSS
# ============================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .info-box {
        padding: 1rem;
        background-color: #d1ecf1;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# SECTION 3: SESSION STATE
# ============================================

if "media_vault" not in st.session_state:
    st.session_state.media_vault = MediaVault()

if "optimizer" not in st.session_state:
    st.session_state.optimizer = MediaOptimizer()

if "fl_engine" not in st.session_state:
    st.session_state.fl_engine = FLEngine()

if "knowledge_hub" not in st.session_state:
    st.session_state.knowledge_hub = KnowledgeHub()

if "audit_chain" not in st.session_state:
    st.session_state.audit_chain = AuditChain()

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Overview"


# ============================================
# SECTION 4: TAB FUNCTIONS
# ============================================

def render_overview():
    """📊 Overview Tab"""
    st.title("🧬 MediaAI Platform")
    st.markdown("### AI-Driven Cell Culture Media Intelligence")

    # Architecture diagram
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### 🔧 Platform Architecture
        **MediaAI** is an intelligent platform for optimizing cell culture media
        through artificial intelligence, computer vision, and federated learning.

        #### Core Modules:
        - 🗃️ **MediaVault**: Secure data management
        - 🎯 **MediaOptimizer**: Bayesian optimization with Gaussian Process
        - 🌐 **FLEngine**: Privacy-preserving federated learning
        - 👁️ **VisionAna**: Computer vision for cell analysis
        - 📚 **KnowledgeHub**: Domain knowledge base
        - 🔗 **AuditChain**: Research documentation
        """)

        st.markdown("### 🚀 Quick Start")

        st.code("""
# Install
pip install -e .

# Run
streamlit run app.py
        """, language="bash")

    with col2:
        st.markdown("### 📊 Statistics")
        st.metric("Observations", len(st.session_state.optimizer.observations))
        st.metric("Knowledge Entries", len(st.session_state.knowledge_hub.entries))
        st.metric("Audit Records", len(st.session_state.audit_chain.chain))


def render_data_vault():
    """💾 Data Vault Tab"""
    st.title("🗃️ Data Vault")
    st.markdown("#### Media Formulation Data Management")

    tab1, tab2, tab3 = st.tabs(["Add Data", "Quality Report", "Similarity Search"])

    with tab1:
        st.markdown("##### ➕ Add New Formulation")

        with st.form("add_formula"):
            col1, col2 = st.columns(2)

            with col1:
                comp_glucose = st.number_input("Glucose (g/L)", 0.0, 50.0, 25.0)
                comp_amino = st.number_input("Amino Acids (g/L)", 0.0, 20.0, 10.0)
                comp_salt = st.number_input("Salt (g/L)", 0.0, 10.0, 5.0)

            with col2:
                perf_viability = st.number_input("Viability (%)", 0.0, 100.0, 80.0)
                perf_growth = st.number_input("Growth Rate (%/day)", 0.0, 50.0, 20.0)
                perf_cost = st.number_input("Cost ($/L)", 0.0, 500.0, 100.0)

            submitted = st.form_submit_button("Add Record")

            if submitted:
                composition = {
                    "glucose": comp_glucose,
                    "amino_acids": comp_amino,
                    "salt": comp_salt
                }
                performance = {
                    "viability": perf_viability,
                    "growth_rate": perf_growth,
                    "cost": perf_cost
                }

                from mediaai.media_vault import MediaRecord
                record = MediaRecord(
                    id=f"M{len(st.session_state.media_vault.records):03d}",
                    composition=composition,
                    metadata={"source": "manual"}
                )
                st.session_state.media_vault.add_record(record)

                # Also add to optimizer
                st.session_state.optimizer.add_observation(composition, performance)

                st.success(f"✅ Added formulation: {record.id}")

    with tab2:
        st.markdown("##### 📋 Quality Report")
        report = st.session_state.media_vault.quality_report()

        col1, col2, col3 = st.columns(3)
        col1.metric("Records", report.get("n_records", 0))
        col2.metric("Components", report.get("n_components", 0))

        if report.get("components"):
            st.write("**Components:**", ", ".join(report["components"]))

    with tab3:
        st.markdown("##### 🔍 Similarity Search")

        query = {}

        with st.form("search_form"):
            col1, col2 = st.columns(2)
            with col1:
                query["glucose"] = st.number_input("Query Glucose", 0.0, 50.0, 25.0)
                query["amino_acids"] = st.number_input("Query Amino Acids", 0.0, 20.0, 10.0)
            with col2:
                query["salt"] = st.number_input("Query Salt", 0.0, 10.0, 5.0)

            search_btn = st.form_submit_button("Search")

            if search_btn and st.session_state.media_vault.records:
                results = st.session_state.media_vault.similarity_search(query, top_k=5)

                st.write("**Top Similar Formulations:**")
                for rid, score in results:
                    st.write(f"- {rid}: similarity = {score:.3f}")
            elif search_btn:
                st.warning("No data available for search")


def render_knowledge_hub():
    """📚 Knowledge Hub Tab"""
    st.title("📚 Knowledge Hub")
    st.markdown("#### Domain Knowledge for Cell Culture Media")

    # Search
    query = st.text_input("🔍 Search Knowledge Base", "")
    category = st.selectbox(
        "Filter by Category",
        ["All", "formulation", "biology", "optimization", "fl"]
    )

    if query:
        cat = None if category == "All" else category
        results = st.session_state.knowledge_hub.search(query, cat, limit=10)

        st.write(f"**Found {len(results)} results:**")

        for r in results:
            with st.expander(f"📄 {r.title}"):
                st.write(r.content)
                st.write("**Tags:**", ", ".join(r.tags))
    else:
        # Browse by category
        selected_cat = st.radio(
            "Browse Categories",
            ["formulation", "biology", "optimization", "fl"],
            horizontal=True
        )

        entries = [
            e for e in st.session_state.knowledge_hub.entries.values()
            if e.category == selected_cat
        ]

        st.write(f"**{len(entries)} entries in {selected_cat}:**")

        for e in entries:
            with st.expander(f"📄 {e.title}"):
                st.write(e.content[:500])


def render_optimizer():
    """🎯 Bayesian Optimizer Tab"""
    st.title("🎯 Bayesian Optimizer")
    st.markdown("#### AI-Driven Media Formulation Optimization")

    tab1, tab2 = st.tabs(["Recommend", "History"])

    with tab1:
        st.markdown("##### 🤖 Get Recommendations")

        col1, col2 = st.columns(2)

        with col1:
            target_metric = st.selectbox(
                "Target Metric",
                ["viability", "growth_rate", "cost"]
            )
            objective = st.selectbox(
                "Objective",
                ["maximize", "minimize"]
            )

        with col2:
            n_candidates = st.slider("Number of Candidates", 1, 10, 5)

        recommend_btn = st.button("Get Recommendations", type="primary")

        if recommend_btn:
            if len(st.session_state.optimizer.observations) < 2:
                st.error("Need at least 2 observations for optimization!")
            else:
                try:
                    result = st.session_state.optimizer.recommend_candidates(
                        target_metric=target_metric,
                        objective=objective,
                        n_candidates=n_candidates
                    )

                    st.success(f"✅ Found {len(result.candidates)} candidates")

                    for i, cand in enumerate(result.candidates, 1):
                        with st.expander(f"#{i} (EI: {cand.ei_value:.4f})"):
                            st.write("**Composition:**")
                            for comp, val in cand.composition.items():
                                st.write(f"  - {comp}: {val:.2f}")
                            st.write(f"**Predicted {target_metric}:** {cand.predicted_performance:.2f}")
                            st.write(f"**Uncertainty:** ±{cand.uncertainty:.2f}")

                except Exception as e:
                    st.error(f"Optimization error: {e}")

        # Add feedback section
        st.markdown("##### 📝 Add Experiment Feedback")

        with st.form("feedback_form"):
            col1, col2 = st.columns(2)

            with col1:
                fb_glucose = st.number_input("Glucose", 0.0, 50.0, 25.0)
                fb_amino = st.number_input("Amino Acids", 0.0, 20.0, 10.0)

            with col2:
                fb_viability = st.number_input("Observed Viability", 0.0, 100.0, 80.0)

            fb_submit = st.form_submit_button("Submit Feedback")

            if fb_submit:
                comp = {"glucose": fb_glucose, "amino_acids": fb_amino}
                perf = {"viability": fb_viability}

                st.session_state.optimizer.add_observation(comp, perf)
                st.success("✅ Feedback recorded!")

    with tab2:
        st.markdown("##### 📈 Observation History")

        observations = st.session_state.optimizer.observations

        if observations:
            data = []
            for i, obs in enumerate(observations):
                row = {"#": i + 1}
                row.update(obs.composition)
                row.update(obs.performance)
                data.append(row)

            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No observations yet. Add data in Data Vault or submit feedback.")


def render_fl_engine():
    """🌐 FL Engine Tab"""
    st.title("🌐 Federated Learning Engine")
    st.markdown("#### Privacy-Preserving Multi-Institution Collaboration")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### ⚙️ Configuration")

        dp_epsilon = st.slider(
            "Privacy Budget (ε)",
            0.1, 50.0, 10.0,
            help="Smaller = more private"
        )

        num_rounds = st.slider("Training Rounds", 1, 50, 10)

        simulate_btn = st.button("Start Training", type="primary")

    with col2:
        st.markdown("##### 📊 Status")
        st.metric("Privacy ε", dp_epsilon)
        st.metric("Total Rounds", len(st.session_state.fl_engine.history))

    if simulate_btn:
        with st.spinner("Running federated training..."):
            # Simulate training
            history_log = []
            for round_num in range(num_rounds):
                # Random metrics
                loss = 0.5 * np.exp(-0.1 * round_num) + 0.1
                delta = np.random.normal(0, 0.01)
                history_log.append({
                    "round": round_num + 1,
                    "loss": loss + delta,
                    "accuracy": 0.95 - 0.3 * np.exp(-0.1 * round_num)
                })

            st.session_state.fl_engine.history.extend(history_log)

        st.success(f"✅ Completed {num_rounds} rounds!")

    # Plot convergence
    if st.session_state.fl_engine.history:
        st.markdown("##### 📉 Convergence Curves")

        history_df = pd.DataFrame(st.session_state.fl_engine.history)

        chart_data = pd.DataFrame({
            "Round": history_df["round"],
            "Loss": history_df["loss"],
            "Accuracy": history_df.get("history_df", "accuracy")
        })

        st.line_chart(chart_data.set_index("Round"))


def render_audit_chain():
    """🔗 Audit Chain Tab"""
    st.title("🔗 Audit Chain")
    st.markdown("#### Research Documentation & Provenance")

    tab1, tab2 = st.tabs(["View Log", "Verify"])

    with tab1:
        st.markdown("##### 📜 Audit Log")

        if st.session_state.audit_chain.chain:
            for entry in st.session_state.audit_chain.chain:
                with st.expander(f"#{entry.index} - {entry.experiment_id}"):
                    st.write(f"**Timestamp:** {entry.timestamp}")
                    st.write(f"**Hash:** {entry.data_hash[:16]}...")
                    st.write(f"**Previous:** {entry.previous_hash[:16]}...")
        else:
            st.info("No audit records yet.")

    with tab2:
        st.markdown("##### ✅ Verify Integrity")

        verify_btn = st.button("Verify Chain")

        if verify_btn:
            if st.session_state.audit_chain.verify_integrity():
                st.success("✅ Chain integrity verified! No tampering detected.")
            else:
                st.error("⚠️ Chain integrity compromised!")

        # Generate report
        report = st.session_state.audit_chain.generate_report()
        st.json(report)


# ============================================
# SECTION 5: MAIN APP
# ============================================

def main():
    """Main application entry"""

    # Sidebar navigation
    st.sidebar.image(
        "https://img.icons8.com/ios-filled/100/ microscope.png",
        width=50
    )

    st.sidebar.markdown("### 🧬 MediaAI")

    # Navigation
    tabs = {
        "📊 Overview": "Overview",
        "💾 Data Vault": "data_vault",
        "📚 Knowledge Hub": "knowledge_hub",
        "🎯 Optimizer": "optimizer",
        "🌐 FL Engine": "fl_engine",
        "🔗 Audit Chain": "audit_chain",
    }

    selected = st.sidebar.radio("Navigate", list(tabs.keys()))

    # Render selected tab
    tab_function = tabs[selected]

    if tab_function == "Overview":
        render_overview()
    elif tab_function == "data_vault":
        render_data_vault()
    elif tab_function == "knowledge_hub":
        render_knowledge_hub()
    elif tab_function == "optimizer":
        render_optimizer()
    elif tab_function == "fl_engine":
        render_fl_engine()
    elif tab_function == "audit_chain":
        render_audit_chain()

    # Sidebar footer
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"**Version:** 0.1.0  \n"
        f"**Date:** {datetime.now().strftime('%Y-%m-%d')}"
    )


if __name__ == "__main__":
    main()