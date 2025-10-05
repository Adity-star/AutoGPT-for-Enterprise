import streamlit as st
import asyncio
from autogpt_core.autogpt_agent import Agent

st.title("AutoGPT for Enterprise")

st.write("Welcome to AutoGPT for Enterprise! This is a Streamlit app that allows you to interact with an AutoGPT agent.")

user_request = st.text_input("Enter your request:")

if st.button("Run Agent"):
    if user_request:
        st.write(f"Running agent with request: {user_request}")
        # Create an agent
        agent = Agent()

        # Run the agent with the user's request
        with st.spinner("Running agent..."):
            results = asyncio.run(agent.run(user_request))
        st.success("Agent finished running.")

        if "market_research_agent" in results and results["market_research_agent"] and "best_business_idea" in results["market_research_agent"]:
            st.subheader("Generated Business Idea:")
            idea = results["market_research_agent"]["best_business_idea"]
            st.write(f"**Idea:** {idea.get('idea', 'N/A')}")
            st.write(f"**Recommendation:** {idea.get('recommendation', 'N/A')}")
            st.write(f"**Final Score:** {idea.get('final_score', 'N/A')}")
            st.json(idea)
        else:
            st.write("No business idea generated or found.")
    else:
        st.write("Please enter a request.")