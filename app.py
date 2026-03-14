"""
Streamlit UI for Multi-Agent Document Intelligence System
A beautiful, interactive interface for analyzing documents with AI agents.
Built with LangGraph for graph-based agent orchestration.
"""

import streamlit as st
import json
import time
from datetime import datetime
import os

# Import our modules (now using LangGraph)
from agents import analyze_document, clear_agent_cache, list_agent_cache
from extract import process_document, calculate_file_hash

# Page configuration
st.set_page_config(
    page_title="🤖 Multi-Agent Document Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --accent-color: #06b6d4;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
        --dark-bg: #0f172a;
        --card-bg: #1e293b;
    }
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        color: #94a3b8;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .agent-card {
        background: linear-gradient(145deg, #1e293b 0%, #334155 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #374151;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .agent-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(99,102,241,0.2);
    }
    
    .agent-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .status-high { background: #fecaca; color: #991b1b; }
    .status-medium { background: #fef3c7; color: #92400e; }
    .status-low { background: #d1fae5; color: #065f46; }
    
    /* Action item cards */
    .action-item {
        background: #1e293b;
        border-left: 4px solid #6366f1;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    /* Risk cards */
    .risk-card {
        background: linear-gradient(145deg, #1e293b 0%, #2d1f3d 100%);
        border-left: 4px solid #ef4444;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    /* Question cards */
    .question-card {
        background: linear-gradient(145deg, #1e293b 0%, #1f2d3d 100%);
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    /* Metrics */
    .metric-container {
        background: linear-gradient(145deg, #1e293b 0%, #374151 100%);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #4b5563;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        color: #94a3b8;
        font-size: 0.875rem;
        margin-top: 0.25rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: #1e293b;
    }
    
    /* Progress animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .processing {
        animation: pulse 2s infinite;
    }
    
    /* File upload area */
    .uploadedFile {
        background: #1e293b !important;
        border: 2px dashed #6366f1 !important;
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)


def render_header():
    """Render the main header."""
    st.markdown('<h1 class="main-header">🤖 Multi-Agent Document Intelligence</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Powered by AI Agents for Deep Document Analysis</p>', unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with file upload and options."""
    with st.sidebar:
        st.markdown("### 📄 Document Input")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload your document",
            type=["pdf", "docx", "txt", "vtt"],
            help="Supported formats: PDF, DOCX, TXT, VTT"
        )
        
        st.markdown("---")
        

        
        # Agent selection
        st.markdown("### 🤖 Agent Configuration")
        
        run_summary = st.checkbox("📝 Summary Agent", value=True)
        run_actions = st.checkbox("✅ Action Extraction Agent", value=True)
        run_risks = st.checkbox("⚠️ Risk Analysis Agent", value=True)
        
        st.markdown("---")
        
        # Cache configuration
        st.markdown("### ⚡ Cache Settings")
        use_cache = st.checkbox(
            "Use cached results", 
            value=True,
            help="If enabled, previously analyzed documents will return cached results instantly"
        )
        force_reanalyze = st.checkbox(
            "Force re-analyze",
            value=False,
            help="Force re-analysis even if cached results exist"
        )
        
        # Clear cache button
        if st.button("🗑️ Clear Cache", use_container_width=True):
            clear_agent_cache()
            st.success("Cache cleared!")
        
        st.markdown("---")
        
        # Process button
        process_btn = st.button(
            "🚀 Analyze Document",
            type="primary",
            use_container_width=True
        )
        
        return uploaded_file, process_btn, {
            "summary": run_summary,
            "actions": run_actions,
            "risks": run_risks,
            "use_cache": use_cache and not force_reanalyze
        }


def render_agent_status(agent_name: str, status: str, icon: str):
    """Render agent processing status."""
    status_colors = {
        "pending": "🔘",
        "processing": "⏳",
        "completed": "✅",
        "error": "❌"
    }
    
    st.markdown(f"""
    <div class="agent-card">
        <div class="agent-title">
            {icon} {agent_name}
            <span style="margin-left: auto;">{status_colors.get(status, "🔘")}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_summary_results(summary_data: dict):
    """Render summary agent results."""
    st.markdown("## 📝 Executive Summary")
    
    summary_text = summary_data.get("summary", "No summary available")
    
    with st.container():
        st.markdown(f"""
        <div class="agent-card">
            <div class="agent-title">📝 Context-Aware Summary Agent</div>
            <p style="color: #94a3b8; font-size: 0.875rem;">
                Processed {summary_data.get('chunk_count', 1)} document chunk(s)
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(summary_text)


def render_action_results(action_data: dict):
    """Render action extraction agent results."""
    st.markdown("## ✅ Action Items & Dependencies")
    
    extracted = action_data.get("extracted_data", {})
    actions = extracted.get("action_items", [])
    
    if not actions:
        st.info("No action items found in the document.")
        return
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value">{len(actions)}</div>
            <div class="metric-label">Total Tasks</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        high_priority = sum(1 for a in actions if a.get("priority", "").lower() == "high")
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value" style="background: linear-gradient(90deg, #ef4444, #f97316); -webkit-background-clip: text;">{high_priority}</div>
            <div class="metric-label">High Priority</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        unassigned = sum(1 for a in actions if a.get("owner", "").lower() in ["unassigned", "not specified", ""])
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value" style="background: linear-gradient(90deg, #f59e0b, #eab308); -webkit-background-clip: text;">{unassigned}</div>
            <div class="metric-label">Unassigned</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        with_deps = sum(1 for a in actions if a.get("dependencies", []))
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-value" style="background: linear-gradient(90deg, #06b6d4, #0ea5e9); -webkit-background-clip: text;">{with_deps}</div>
            <div class="metric-label">With Dependencies</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Action items list
    for action in actions:
        priority = action.get("priority", "Medium").lower()
        priority_color = {"high": "#ef4444", "medium": "#f59e0b", "low": "#10b981"}.get(priority, "#6366f1")
        
        with st.expander(f"**{action.get('id', '?')}. {action.get('task', 'No description')}**", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**👤 Owner:** {action.get('owner', 'Unassigned')}")
                st.markdown(f"**📅 Deadline:** {action.get('deadline', 'Not specified')}")
            
            with col2:
                st.markdown(f"**🎯 Priority:** :{priority}[{action.get('priority', 'Medium')}]")
                st.markdown(f"**📊 Status:** {action.get('status', 'Pending')}")
            
            if action.get("dependencies"):
                st.markdown(f"**🔗 Dependencies:** {', '.join(str(d) for d in action.get('dependencies', []))}")
            
            if action.get("source_context"):
                st.markdown(f"*Source: \"{action.get('source_context')}\"*")


def render_risk_results(risk_data: dict):
    """Render risk analysis agent results."""
    st.markdown("## ⚠️ Risks & Open Issues")
    
    analysis = risk_data.get("analysis", {})
    
    # Overall risk score
    risk_score = analysis.get("risk_score", {})
    overall = risk_score.get("overall", "Unknown")
    
    score_colors = {"high": "#ef4444", "medium": "#f59e0b", "low": "#10b981"}
    score_color = score_colors.get(overall.lower(), "#6366f1")
    
    st.markdown(f"""
    <div class="agent-card" style="border-color: {score_color};">
        <div class="agent-title">
            🎯 Overall Risk Assessment: 
            <span style="color: {score_color}; margin-left: 0.5rem;">{overall.upper()}</span>
        </div>
        <p style="color: #94a3b8;">{risk_score.get('rationale', 'No rationale provided')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs for different categories
    tab1, tab2, tab3, tab4 = st.tabs(["❓ Open Questions", "⚡ Risks", "💭 Assumptions", "📋 Missing Info"])
    
    with tab1:
        questions = analysis.get("open_questions", [])
        if questions:
            for q in questions:
                st.markdown(f"""
                <div class="question-card">
                    <strong>Q{q.get('id', '?')}: {q.get('question', 'No question')}</strong>
                    <p style="color: #94a3b8; margin: 0.5rem 0;">Impact: {q.get('impact', 'Unknown')}</p>
                    <p style="color: #6366f1;">💡 Suggested: {q.get('suggested_resolution', 'No suggestion')}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No open questions identified.")
    
    with tab2:
        risks = analysis.get("risks", [])
        if risks:
            for r in risks:
                likelihood = r.get("likelihood", "Unknown").lower()
                impact = r.get("impact", "Unknown").lower()
                
                st.markdown(f"""
                <div class="risk-card">
                    <strong>Risk {r.get('id', '?')}: {r.get('risk', 'No description')}</strong>
                    <p style="color: #94a3b8; margin: 0.5rem 0;">
                        Category: {r.get('category', 'Unknown')} | 
                        Likelihood: {r.get('likelihood', 'Unknown')} | 
                        Impact: {r.get('impact', 'Unknown')}
                    </p>
                    <p style="color: #10b981;">🛡️ Mitigation: {r.get('mitigation', 'No mitigation suggested')}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No significant risks identified.")
    
    with tab3:
        assumptions = analysis.get("assumptions", [])
        if assumptions:
            for a in assumptions:
                with st.expander(f"Assumption {a.get('id', '?')}: {a.get('assumption', 'No assumption')[:50]}..."):
                    st.markdown(f"**Assumption:** {a.get('assumption', 'N/A')}")
                    st.markdown(f"**Risk if Wrong:** {a.get('risk_if_wrong', 'N/A')}")
                    st.markdown(f"**Validation Needed:** {a.get('validation_needed', 'N/A')}")
        else:
            st.info("No assumptions identified.")
    
    with tab4:
        missing = analysis.get("missing_information", [])
        if missing:
            for m in missing:
                importance_color = {"critical": "🔴", "important": "🟡", "nice-to-have": "🟢"}.get(
                    m.get("importance", "").lower(), "⚪"
                )
                st.markdown(f"""
                {importance_color} **{m.get('description', 'No description')}**
                - Importance: {m.get('importance', 'Unknown')}
                - Impact: {m.get('impact_if_not_addressed', 'Unknown')}
                """)
        else:
            st.success("No critical missing information identified.")


def render_json_export(results: dict):
    """Render JSON export option."""
    st.markdown("---")
    st.markdown("## 📥 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # JSON download
        json_str = json.dumps(results, indent=2, default=str)
        st.download_button(
            label="📄 Download Full Report (JSON)",
            data=json_str,
            file_name=f"document_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col2:
        # Show raw JSON
        with st.expander("View Raw JSON"):
            st.json(results.get("consolidated_report", results))


def main():
    """Main application entry point."""
    render_header()
    
    # Sidebar inputs
    uploaded_file, process_btn, agent_config = render_sidebar()
    
    # Main content area
    if process_btn:
        document_text = None
        file_info = None
        
        # Get document text
        if uploaded_file is not None:
            with st.spinner("📄 Extracting text from document..."):
                try:
                    file_bytes = uploaded_file.read()
                    document_text = process_document(file_bytes, filename=uploaded_file.name)
                    file_info = {
                        "name": uploaded_file.name,
                        "size": len(file_bytes),
                        "hash": calculate_file_hash(file_bytes)
                    }
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    return
        else:
            st.warning("⚠️ Please upload a document to analyze.")
            return
        
        if not document_text or len(document_text.strip()) < 50:
            st.error("❌ Document is too short or empty. Please provide a longer document (500+ words recommended).")
            return
        
        # Show file info
        if file_info:
            st.markdown(f"""
            <div class="agent-card">
                <div class="agent-title">📄 Document Information</div>
                <p><strong>Name:</strong> {file_info['name']}</p>
                <p><strong>Size:</strong> {file_info['size']:,} bytes</p>
                <p><strong>Word Count:</strong> ~{len(document_text.split()):,} words</p>
                <p style="font-size: 0.75rem; color: #64748b;">Hash: {file_info['hash'][:16]}...</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(message: str, progress: float):
            progress_bar.progress(progress)
            status_text.markdown(f"**{message}**")
        
        # Run analysis (with caching support)
        with st.spinner("🤖 Agents are analyzing your document..."):
            try:
                results = analyze_document(
                    document_text, 
                    progress_callback=update_progress,
                    use_cache=agent_config.get("use_cache", True)
                )
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Success message with cache indicator
                if results.get("from_cache", False):
                    st.success("⚡ Analysis retrieved from cache (instant!)")
                    st.info(f"🔑 Content hash: {results.get('content_hash', 'N/A')[:32]}...")
                else:
                    st.success("✅ Analysis complete! Results cached for future use.")
                
                # Render results
                st.markdown("---")
                
                # Summary section
                if agent_config["summary"]:
                    render_summary_results(results["agents_results"].get("summary", {}))
                    st.markdown("---")
                
                # Actions section
                if agent_config["actions"]:
                    render_action_results(results["agents_results"].get("actions", {}))
                    st.markdown("---")
                
                # Risks section
                if agent_config["risks"]:
                    render_risk_results(results["agents_results"].get("risks", {}))
                
                # Export options
                render_json_export(results)
                
            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.exception(e)
    
    else:
        # Welcome message when no document is loaded
        st.markdown("""
        <div class="agent-card" style="text-align: center; padding: 3rem;">
            <h2 style="color: #e2e8f0;">Welcome to Document Intelligence</h2>
            <p style="color: #94a3b8; font-size: 1.1rem; max-width: 600px; margin: 1rem auto;">
                Upload a document or paste text in the sidebar to begin analysis. 
                Our AI agents will extract summaries, action items, and identify risks.
            </p>
            <p style="color: #6366f1; font-size: 0.9rem;">
                ⚡ Powered by <strong>LangGraph</strong> for graph-based agent orchestration
            </p>
            <br>
            <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
                <div>
                    <span style="font-size: 2rem;">📝</span>
                    <p style="color: #94a3b8;">Summary Agent</p>
                </div>
                <div>
                    <span style="font-size: 2rem;">✅</span>
                    <p style="color: #94a3b8;">Action Extraction</p>
                </div>
                <div>
                    <span style="font-size: 2rem;">⚠️</span>
                    <p style="color: #94a3b8;">Risk Analysis</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        
        # Feature highlights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="agent-card">
                <div class="agent-title">📄 Supported Formats</div>
                <ul style="color: #94a3b8;">
                    <li>PDF Documents</li>
                    <li>Word Documents (.docx)</li>
                    <li>Text Files (.txt)</li>
                    <li>Subtitles (.vtt)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="agent-card">
                <div class="agent-title">🎯 Use Cases</div>
                <ul style="color: #94a3b8;">
                    <li>Meeting Transcripts</li>
                    <li>Legal Documents</li>
                    <li>Project Briefs</li>
                    <li>Policy Documents</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="agent-card">
                <div class="agent-title">✨ Features</div>
                <ul style="color: #94a3b8;">
                    <li>Context-Aware Summaries</li>
                    <li>Structured Action Items</li>
                    <li>Risk Assessment</li>
                    <li>JSON Export</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
