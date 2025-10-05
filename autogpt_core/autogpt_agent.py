from autogpt_core.planner import AgentPlanner
from autogpt_core.modules.market_researcher.graph import get_or_generate_market_research_idea
from autogpt_core.modules.market_researcher.state import MarketResearchState, AnalysisConfig
from autogpt_core.modules.landing_page_builder.landing_page_graph import landing_page_graph
from autogpt_core.modules.email_campaign.campaign_manager import email_campaign_graph
from autogpt_core.modules.blog_writer.blog_generator import blog_generator_graph

class Agent:
    def __init__(self):
        self.planner = AgentPlanner()
        self.available_agents = {
            "market_research_agent": self.run_market_research,
            "landing_page_agent": self.run_landing_page_creation,
            "email_campaign_agent": self.run_email_campaign,
            "blog_writer_agent": self.run_blog_writer,
        }

    async def run(self, user_request: str):
        """
        Runs the agent to fulfill the user's request.

        Args:
            user_request: The user's request.
        """
        print(f"Received request: {user_request}")
        plan = self.planner.plan(user_request)
        print(f"Generated plan steps: {plan.steps}")
        
        results = {}
        for step in plan.steps:
            agent_name = step.agent
            if agent_name in self.available_agents:
                agent_function = self.available_agents[agent_name]
                result = await agent_function(step.params)
                results[agent_name] = result
            else:
                print(f"Unknown agent: {agent_name}")
        return results

    async def run_market_research(self, params: dict):
        """
        Runs the market research agent.

        Args:
            params: The parameters for the market research agent.
        """
        print("Running market research agent...")
        # Create a MarketResearchState object from the params dictionary
        config = AnalysisConfig(
            max_retries=params.get("max_retries", 2),
            batch_size=params.get("batch_size", 3),
            timeout=params.get("timeout", 30),
            enable_caching=params.get("enable_caching", True),
            cache_ttl_minutes=params.get("cache_ttl_minutes", 60)
        )
        state = MarketResearchState(config=config, user_idea=params.get("user_idea"))
        return await get_or_generate_market_research_idea(state)

    async def run_landing_page_creation(self, params: dict):
        """
        Runs the landing page creation agent.

        Args:
            params: The parameters for the landing page creation agent.
        """
        print("Running landing page creation agent...")
        graph = landing_page_graph()
        await graph.ainvoke(params)

    async def run_email_campaign(self, params: dict):
        """
        Runs the email campaign agent.

        Args:
            params: The parameters for the email campaign agent.
        """
        print("Running email campaign agent...")
        graph = email_campaign_graph()
        await graph.ainvoke(params)

    async def run_blog_writer(self, params: dict):
        """
        Runs the blog writer agent.

        Args:
            params: The parameters for the blog writer agent.
        """
        print("Running blog writer agent...")
        graph = blog_generator_graph()
        await graph.ainvoke(params)