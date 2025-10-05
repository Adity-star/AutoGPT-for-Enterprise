from pydantic import BaseModel
from typing import List, Dict
import json
from autogpt_core.core.llm_service import LLMService

class Step(BaseModel):
    step_id: int
    agent: str
    params: Dict
    depends_on: List[int] = []

class Plan(BaseModel):
    steps: List[Step]

class AgentPlanner:
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

        Return ONLY a JSON workflow plan as a list of steps. Only include agents that are strictly necessary to fulfill the user's request. DO NOT include agents that are not explicitly required by the user's request. Do not include any other text or explanation.

        Thought: Analyze the user's request and determine which of the available agents are absolutely essential to complete the task. Consider if any agents are implicitly required (e.g., market research for a startup idea). Construct a plan with only the necessary steps.

        Each step must have:
        - step_id (int, starting from 1)
        - agent (string, one of the agents above)
        - params (dict with relevant inputs)
        - depends_on (list of step_ids this step depends on)

        Example for 'give me an idea for building startup in 2025':
        ```json
        [
          {{"step_id": 1, "agent": "market_research_agent", "params": {{"query": "startup ideas for 2025"}}, "depends_on": []}}
        ]
        ```

        Example for 'build a landing page for my new AI SaaS idea':
        ```json
        [
          {{"step_id": 1, "agent": "landing_page_agent", "params": {{"idea": "AI SaaS for personalized learning"}}, "depends_on": []}}
        ]
        ```
        """

        raw_output = self.llm.sync_chat(prompt)
        
        # Extract JSON from the raw output using regex
        import re
        json_match = re.search(r"```json\n([\s\S]*?)\n```", raw_output)
        if json_match:
            json_str = json_match.group(1)
        else:
            # If no code block, try to parse the whole output as JSON
            json_str = raw_output

        try:
            steps_data = json.loads(json_str)
            steps = [Step(**s) for s in steps_data]
            return Plan(steps=steps)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON: {json_str}. Original output: {raw_output}") from e
        except Exception as e:
            raise ValueError(f"LLM returned invalid plan: {raw_output}") from e