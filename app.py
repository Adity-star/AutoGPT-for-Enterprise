import streamlit as st
import requests
from requests.exceptions import RequestException
import json
import time

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


mode = st.radio("choose what you want to do:", ["🔍 Generate New Idea", "✅ Validate My Idea"])

user_idea= None
if mode == "✅ Validate My Idea" and not user_idea:
    st.warning("Please enter your idea below before running the analysis.")
    user_idea = st.text_area("Enter your business idea", height=100, placeholder="e.g. AI-powered personal finance assistant")

payload = {}
if user_idea:
    payload["user_idea"] = user_idea
# Add a run button
if st.button("Run Market Research", type="primary"):
    with st.spinner("Running market research... This may take a few minutes."):
        try:
            response = requests.post(
                "http://localhost:8000/api/research",
                json=payload,  # Empty payload uses default config
                timeout=300  # 5 minute timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if result["status"] == "success":
                    st.success("✅ Market research completed!")
                    
                    # Get the best idea data
                    best_idea = result.get('best_business_idea', {})
                    
                    # Display the best idea title
                    st.markdown("### 🏆 Best Business Idea")
                    st.markdown(f"**{best_idea.get('idea', 'No idea generated')}**")
                    
                    # Create columns for scores
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
                    
                    # Display detailed analysis
                    st.markdown("### 📈 Detailed Analysis")
                    
                    # Demand Analysis
                    with st.expander("📊 Demand Analysis", expanded=True):
                        st.markdown(f"""
                        <div class='analysis-box'>
                            {best_idea.get('demand_analysis', 'No demand analysis available')}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Competition Analysis
                    with st.expander("🎯 Competition Analysis", expanded=True):
                        st.markdown(f"""
                        <div class='analysis-box'>
                            {best_idea.get('competition_analysis', 'No competition analysis available')}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Economics Analysis
                    with st.expander("💰 Economics Analysis", expanded=True):
                        st.markdown(f"""
                        <div class='analysis-box'>
                            {best_idea.get('unit_economics', 'No economics analysis available')}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Add download button for results
                    st.download_button(
                        label="📥 Download Full Analysis",
                        data=json.dumps(best_idea, indent=2),
                        file_name="market_research_analysis.json",
                        mime="application/json"
                    )
                else:
                    st.error(f"Error: {result.get('error', 'Unknown error')}")
            else:
                st.error(f"API error: {response.status_code} - {response.text}")
        except RequestException as e:
            st.error(f"Network error: {e}")
            st.info("Make sure the backend server is running on http://localhost:8000")

# Add footer
st.markdown("---")
st.markdown("Made with ❤️ by Aditya Ak")
