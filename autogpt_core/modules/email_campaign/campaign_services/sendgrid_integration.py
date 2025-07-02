import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from autogpt_core.utils.logger import logger
from autogpt_core.modules.email_campaign.memory_logs.campaign_memory import log_campaign_email
from autogpt_core.modules.email_campaign.schema import CampaignState


from sendgrid.helpers.mail import Mail, Email, To, Content

async def send_email_with_sendgrid_async(state: CampaignState) -> CampaignState:
    logger.info("Sending email using SendGrid...")

    payload = state.email_payload  
    if not payload:
        raise ValueError("Email payload is missing from state.")

    # Build plain text version by stripping HTML tags or generating separately
    plain_text_content = (
        "Hi {name},\n\n"
        "As a founder or owner of an early-stage startup or small business, managing your finances can feel overwhelming. "
        "Our AI-powered personal finance assistant, {product}, helps you manage your financials with ease and confidence.\n\n"
        "With {product}, you can:\n"
        "- Track and categorize expenses easily\n"
        "- Gain actionable insights to make informed decisions\n"
        "- Get personalized recommendations to optimize your finances\n\n"
        "If you’re interested, I’d be happy to schedule a quick call.\n\n"
        "Best regards,\n{sender}"
    ).format(
        name=payload.get("recipient_name", "there"),
        product=payload.get("product_name", "Your SaaS Product"),
        sender=payload.get("sender_name", "Your Name"),
    )

    message = Mail(
        from_email=Email(payload["from_email"]),
        to_emails=To(payload["to_email"]),
        subject=payload["subject"],
        plain_text_content=Content("text/plain", plain_text_content),
        html_content=Content("text/html", payload["content"]),
    )

    try:
        sg = SendGridAPIClient(os.getenv("SENDGRID_API_KEY"))
        response = sg.send(message)

        logger.info(f"Email sent: Status Code {response.status_code}")
        send_status = "sent" if response.status_code == 202 else "failed"
    except Exception as e:
        logger.error(f"Failed to send email: {str(e)}")
        send_status = "error"

    await log_campaign_email(
        to_email=payload["to_email"],
        subject=payload["subject"],
        content=payload["content"],
        delivery_status=send_status,
        idea_reference=state.idea.get("id") if state.idea else None
    )

    return state.copy(update={"send_status": send_status})

