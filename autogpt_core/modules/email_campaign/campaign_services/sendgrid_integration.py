#  Calls SendGrid API to send

from sendgrid.helpers.mail import Mail
from autogpt_core.modules.email_campaign.schema import CampaignState

import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from autogpt_core.utils.logger import logger
from autogpt_core.modules.email_campaign.memory_logs.campaign_memory import log_campaign_email


async def send_email_with_sendgrid_async(state: dict) -> dict:
    logger.info("Sending email using SendGrid...")

    payload = state.get("email_payload")
    if not payload:
        raise ValueError("Email payload is missing from state.")

    try:
        message = Mail(
            from_email=payload["from_email"],
            to_emails=payload["to_email"],
            subject=payload["subject"],
            plain_text_content=payload["content"]
        )

        sg = SendGridAPIClient(api_key=os.getenv("SENDGRID_API_KEY"))
        response = sg.send(message)

        logger.info(f"Email sent: Status Code {response.status_code}")
        send_status = "sent" if response.status_code == 202 else "failed"
    except Exception as e:
        logger.error(f"Failed to send email: {str(e)}")
        send_status = "error"

    # Log the send result asynchronously
    await log_campaign_email(
        to_email=payload["to_email"],
        subject=payload["subject"],
        content=payload["content"],
        delivery_status=send_status,
        idea_reference=state.get("idea_data", {}).get("id")
    )

    return {
        **state,
        "send_status": send_status
    }


