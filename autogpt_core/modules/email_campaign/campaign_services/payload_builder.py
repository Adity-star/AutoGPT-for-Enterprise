# Formats SendGrid-ready payload

from autogpt_core.utils.logger import logger
from autogpt_core.modules.email_campaign.schema import CampaignState


def build_email_payload(state: CampaignState) -> CampaignState:
    logger.info("Building email payload...")

    email_content_html = f"""
    <html>
    <body>
        <p>Hi {state.contacts[0] if state.contacts else 'there'},</p>
        <p>As a founder or owner of an early-stage startup or small business, managing your finances can feel overwhelming.</p>
        <p>Our AI-powered personal finance assistant helps you manage your financials with ease and confidence.</p>
        <ul>
            <li>Track and categorize expenses easily</li>
            <li>Gain actionable insights to make informed decisions</li>
            <li>Get personalized recommendations to optimize your finances</li>
        </ul>
        <p>If you’re interested, I’d be happy to schedule a quick call.</p>
        <p>Best regards,<br>Your Name</p>
    </body>
    </html>
    """

    idea_data = state.idea or {}
    sender_email = "aakuskar.980@gmail.com"
    recipient_email = "adityaakuskar123@gmail.com"
    subject = f"{idea_data.get('idea', 'AI Hiring Assistant')} – Make Recruiting Easier"

    payload = {
        "from_email": sender_email,
        "to_email": recipient_email,
        "subject": subject,
        "content": email_content_html,
        "recipient_name": recipient_email.split("@")[0],  
        "sender_name": "Aditya Akuskar",
        "product_name": idea_data.get('idea', 'Your SaaS Product')
    }

    logger.info(f"Email payload built successfully: {subject} -> recipient: {recipient_email}")

    state_data = state.dict()
    state_data["email_payload"] = payload

    return CampaignState(**state_data)


