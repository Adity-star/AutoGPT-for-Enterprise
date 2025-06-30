# Loads audience from DB/memory

from autogpt_core.modules.email_campaign.memory_logs.campaign_memory import init_email_log_db
from autogpt_core.utils.logger import logger
import sqlite3
from datetime import datetime

DB_FILE = "data/email_campaign_logs.db"

def log_email_campaign_result(state: dict) -> dict:
    logger.info("Logging email campaign result...")

    payload = state.get("email_payload")
    status = state.get("delivery_status")
    idea_data = state.get("idea_data")

    if not payload:
        logger.warning("No email payload to log.")
        return state

    init_email_log_db()

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("""
        INSERT INTO campaign_logs (
            to_email, subject, content, delivery_status, idea_reference, sent_at
        ) VALUES (?, ?, ?, ?, ?, ?)
    """, (
        payload.get("to_email"),
        payload.get("subject"),
        payload.get("content"),
        status,
        idea_data.get("idea") if idea_data else None,
        datetime.utcnow().isoformat()
    ))

    conn.commit()
    conn.close()

    logger.info("Campaign result logged successfully.")
    return state