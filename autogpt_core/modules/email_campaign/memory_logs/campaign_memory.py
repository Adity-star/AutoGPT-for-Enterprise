import aiosqlite
from datetime import datetime
from autogpt_core.utils.logger import logger

DB_FILE = "email_campaign_logs.db"

async def log_campaign_email(to_email: str, subject: str, content: str, delivery_status: str, idea_reference: str = None):
    try:
        async with aiosqlite.connect(DB_FILE) as db:
            sent_at = datetime.utcnow().isoformat()
            await db.execute("""
                INSERT INTO campaign_logs (to_email, subject, content, delivery_status, idea_reference, sent_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (to_email, subject, content, delivery_status, idea_reference, sent_at))
            await db.commit()
    except Exception as e:
        logger.error(f"Failed to log email campaign async: {e}")
