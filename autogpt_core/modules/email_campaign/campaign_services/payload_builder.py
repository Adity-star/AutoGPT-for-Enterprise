# Formats SendGrid-ready payload

from autogpt_core.utils.logger import logger
from autogpt_core.modules.email_campaign.schema import CampaignState


def build_email_payload(state: CampaignState) -> CampaignState:
    logger.info("Building email payload...")

    email_content = state.email_content
    idea_data = state.idea or {}
    sender_email = "aakuskar.980@gmail.com"
    recipient_email = "adityaakuskar123@gmail.com"  # fixed recipient
    subject = f"{idea_data.get('idea', 'AI Hiring Assistant')} – Make Recruiting Easier"

    if not email_content:
        raise ValueError("Email content is missing in state.")

    payload = {
        "from_email": sender_email,
        "to_email": recipient_email,
        "subject": subject,
        "content": email_content
    }

    logger.info(f"Email payload built successfully: {subject} -> recipient: {recipient_email}")

    state_data = state.dict()
    state_data["email_payload"] = payload

    return CampaignState(**state_data)

