from langgraph.graph import StateGraph
from langchain_core.runnables import RunnableLambda, Runnable
from autogpt_core.modules.email_campaign.schema import CampaignState

from autogpt_core.modules.email_campaign.campaign_services.email_gen import generate_email_copy
from autogpt_core.modules.email_campaign.campaign_services.payload_builder import build_email_payload
from autogpt_core.modules.email_campaign.campaign_services.sendgrid_integration import send_email_with_sendgrid_async
from autogpt_core.modules.email_campaign.campaign_services.email_monitoring import load_campaign_content


def email_campaign_graph():
    graph = StateGraph(CampaignState)

    graph.add_node("load_campaign_content", RunnableLambda(load_campaign_content))  
    graph.add_node("generate_email_copy", RunnableLambda(generate_email_copy))     
    graph.add_node("build_email_payload", RunnableLambda(build_email_payload))      
    graph.add_node("send_email", RunnableLambda(send_email_with_sendgrid_async))   

    graph.set_entry_point("load_campaign_content")
    graph.set_finish_point("send_email")

    graph.add_edge("load_campaign_content", "generate_email_copy")
    graph.add_edge("generate_email_copy", "build_email_payload")
    graph.add_edge("build_email_payload", "send_email")

    return graph.compile()
