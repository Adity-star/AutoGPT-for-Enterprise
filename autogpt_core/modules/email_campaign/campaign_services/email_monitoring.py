# Loads audience from DB/memory

from autogpt_core.modules.email_campaign.memory_logs.campaign_memory import log_campaign_email
from autogpt_core.utils.logger import logger
from autogpt_core.modules.email_campaign.schema import CampaignState

#DB_FILE = "data/email_campaign_logs.db"

def log_email_campaign_result(state: CampaignState) -> CampaignState:
    logger.info("Logging email campaign result...")

    payload = state.email_payload
    status = state.send_status
    idea_data = state.idea

    if not payload:
        logger.warning("No email payload to log.")
        return state

    # Assume log_campaign_email() is synchronous here; adjust if async
    log_campaign_email(
        to_email=payload.get("to_email"),
        subject=payload.get("subject"),
        content=payload.get("content"),
        delivery_status=status,
        idea_reference=idea_data.get("idea") if idea_data else None
    )

    logger.info("Campaign result logged successfully.")
    return state