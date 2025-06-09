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
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🚀 Market Research Agent")
st.markdown("""
    This tool helps you analyze market trends and generate business ideas based on your industry and keywords.
    Enter your industry and keywords below to get started.
""")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    industry = st.text_input(
        "Industry",
        value="Tech",
        help="Enter the industry you want to research"
    )

with col2:
    keywords = st.text_area(
        "Keywords",
        value="AI, automation, analytics",
        help="Enter keywords separated by commas"
    )

# Add a run button
if st.button("Run Market Research", type="primary"):
    if not industry or not keywords:
        st.error("Please fill in both industry and keywords fields")
    else:
        payload = {
            "industry": industry,
            "keywords": [kw.strip() for kw in keywords.split(",") if kw.strip()]
        }
        
        with st.spinner("Running market research... This may take a few minutes."):
            try:
                response = requests.post(
                    "http://localhost:8000/api/research",
                    json=payload,
                    timeout=300  # 5 minute timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("✅ Market research completed!")
                    
                    # Display results in expandable sections
                    with st.expander("📊 Market Analysis", expanded=True):
                        st.json(result["data"])
                    
                    # Add download button for results
                    st.download_button(
                        label="📥 Download Results",
                        data=json.dumps(result["data"], indent=2),
                        file_name="market_research_results.json",
                        mime="application/json"
                    )
                else:
                    st.error(f"API error: {response.status_code} - {response.text}")
            except RequestException as e:
                st.error(f"Network error: {e}")
                st.info("Make sure the backend server is running on http://localhost:8000")

# Add footer
st.markdown("---")
st.markdown("Made with ❤️ by Aditya Ak")
