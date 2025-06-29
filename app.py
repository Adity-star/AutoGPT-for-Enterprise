import streamlit as st
import requests
from requests.exceptions import RequestException
import json
from autogpt_core.utils.idea_memory import load_ideas_from_db

# Configure the page
st.set_page_config(
    page_title="Market Research Agent",
    page_icon="🚀",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 5px;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
    .stTextArea>div>div>textarea {
        border-radius: 5px;
    }
    .analysis-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .score-box {
        background-color: #e6f3ff;
        padding: 15px;
        border-radius: 8px;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🚀 Market Research Agent")
st.markdown("""
This tool helps you either **generate new business ideas** or **validate your existing startup idea** by analyzing demand, competition, and viability.
""")

mode = st.radio("Choose what you want to do:", ["🔍 Generate New Idea", "✅ Validate My Idea"])

user_idea = None
if mode == "✅ Validate My Idea":
    user_idea = st.text_area("Enter your business idea", height=100, placeholder="e.g. AI-powered personal finance assistant")
    if not user_idea:
        st.warning("Please enter your idea below before running the analysis.")

payload = {}
if user_idea:
    payload["user_idea"] = user_idea

if st.button("Run Market Research", type="primary"):
    with st.spinner("Running market research... This may take a few minutes."):
        try:
            response = requests.post(
                "http://localhost:8000/api/research",
                json=payload,  # Empty payload triggers new idea generation
                timeout=300  # 5 minute timeout
            )
            if response.status_code == 200:
                result = response.json()

                if result["status"] == "success":
                    st.success("✅ Market research completed!")

                    # Updated key here: 'best_idea' instead of 'best_business_idea'
                    best_idea = result.get('best_idea', {})

                    st.markdown("### 🏆 Best Business Idea")
                    st.markdown(f"**{best_idea.get('idea', 'No idea generated')}**")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("#### 📊 Final Score")
                        st.markdown(f"**{best_idea.get('final_score', 'N/A')}/100**")
                    with col2:
                        st.markdown("#### ✅ Validation Score")
                        st.markdown(f"**{best_idea.get('validation_score', 'N/A')}/10**")
                    with col3:
                        st.markdown("#### 💡 Recommendation")
                        st.markdown(f"**{best_idea.get('recommendation', 'N/A')}**")

                    st.markdown("### 📈 Detailed Analysis")

                    with st.expander("📊 Demand Analysis", expanded=True):
                        st.markdown(f"""
                        <div class='analysis-box'>
                            {best_idea.get('demand_analysis', 'No demand analysis available')}
                        </div>
                        """, unsafe_allow_html=True)

                    with st.expander("🎯 Competition Analysis", expanded=True):
                        st.markdown(f"""
                        <div class='analysis-box'>
                            {best_idea.get('competition_analysis', 'No competition analysis available')}
                        </div>
                        """, unsafe_allow_html=True)

                    with st.expander("💰 Economics Analysis", expanded=True):
                        st.markdown(f"""
                        <div class='analysis-box'>
                            {best_idea.get('unit_economics', 'No economics analysis available')}
                        </div>
                        """, unsafe_allow_html=True)

                    st.download_button(
                        label="📥 Download Full Analysis",
                        data=json.dumps(best_idea, indent=2),
                        file_name="market_research_analysis.json",
                        mime="application/json"
                    )

                    # Show latest validated idea from short-term memory DB
                    st.markdown("---")
                    st.markdown("### 🧠 Short-Term Memory: Last Validated Idea")
                    latest_ideas = load_ideas_from_db(limit=1)
                    if latest_ideas:
                        idea = latest_ideas[0]
                        st.markdown(f"**Idea:** {idea.get('idea', '')}")
                        st.markdown(f"**Trend Score:** {idea.get('trend_score', '')}")
                        st.markdown(f"**Demand Analysis:** {idea.get('demand_analysis', '')}")
                        st.markdown(f"**Demand Score:** {idea.get('demand_score', '')}")
                        st.markdown(f"**Competition Analysis:** {idea.get('competition_analysis', '')}")
                        st.markdown(f"**Competition Score:** {idea.get('competition_score', '')}")
                        st.markdown(f"**Unit Economics:** {idea.get('unit_economics', '')}")
                        st.markdown(f"**Economics Score:** {idea.get('economics_score', '')}")
                        st.markdown(f"**Final Score:** {idea.get('final_score', '')}")
                        st.markdown(f"**Scoring Breakdown:** {idea.get('scoring_breakdown', '')}")
                        st.markdown(f"**Validation Score:** {idea.get('validation_score', '')}")
                        st.markdown(f"**Recommendation:** {idea.get('recommendation', '')}")
                        st.markdown(f"**Validation Summary:** {idea.get('validation_summary', '')}")
                        st.markdown(f"**Created At:** {idea.get('created_at', '')}")
                    else:
                        st.info("No validated idea in short-term memory yet.")
                else:
                    st.error(f"Error: {result.get('error', 'Unknown error')}")
            else:
                st.error(f"API error: {response.status_code} - {response.text}")
        except RequestException as e:
            st.error(f"Network error: {e}")
            st.info("Make sure the backend server is running on http://localhost:8000")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Aditya Ak")
