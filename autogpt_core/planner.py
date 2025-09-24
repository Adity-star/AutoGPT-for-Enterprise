# core/planner_agent/planner.py

from pydantic import BaseModel
from typing import List, Dict, Optional
import json

class Step(BaseModel):
    step_id: int
    agent: str
    params: Dict
    depends_on: List[int] = []

class Plan(BaseModel):
    steps: List[Step]


# core/planner_agent/task_planner_llm.py

from autogpt_core.core.llm_service import LLMService

class LLMTaskPlanner:
    def __init__(self):
        self.llm = LLMService()

    def plan(self, query: str) -> Plan:
        """
        Use LLM to dynamically generate an ordered workflow.
        """
        prompt = f"""
        You are a workflow planner for an autonomous business AI.
        The user request is: "{query}"

        Available agents:
        1. market_research_agent - does market/competitor analysis
        2. landing_page_agent - builds landing pages
        3. email_campaign_agent - creates & sends email campaigns
        4. blog_writer_agent - writes blogs/articles

        Return a JSON workflow plan as a list of steps.
        Each step must have:
        - step_id (int, starting from 1)
        - agent (string, one of the agents above)
        - params (dict with relevant inputs)
        - depends_on (list of step_ids this step depends on)

        Example:
        [
          {{"step_id": 1, "agent": "market_research_agent", "params": {{"industry": "AI SaaS"}}, "depends_on": []}},
          {{"step_id": 2, "agent": "blog_writer_agent", "params": {{"topic": "Competitor Analysis"}}, "depends_on": [1]}}
        ]
        """

        raw_output = self.llm.generate(prompt)
        try:
            steps_data = json.loads(raw_output)
            steps = [Step(**s) for s in steps_data]
            return Plan(steps=steps)
        except Exception as e:
            raise ValueError(f"LLM returned invalid plan: {raw_output}") from e

# core/planner_agent/planner.py

class AgentPlanner:
    def __init__(self):
        self.llm_planner = LLMTaskPlanner()

    def plan(self, query: str) -> Plan:
        """
        Hybrid: try simple rules first, fallback to LLM for complex queries.
        """
        query_lower = query.lower()

        # Simple heuristic (fast path)
        if "landing page" in query_lower and "blog" in query_lower:
            return Plan(steps=[
                Step(step_id=1, agent="landing_page_agent", params={"product": "AI SaaS"}, depends_on=[]),
                Step(step_id=2, agent="blog_writer_agent", params={"topic": "Product Launch Blog"}, depends_on=[1])
            ])

        # Fallback → LLM-powered planner
        return self.llm_planner.plan(query)
