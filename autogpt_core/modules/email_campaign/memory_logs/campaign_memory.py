import os
import aiosqlite
import sqlite3
from datetime import datetime
from autogpt_core.utils.logger import logger

DB_FILE = "data/email_campaign_logs.db"
os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)

def init_db():
    """Initialize the database and create the campaign_logs table if it does not exist."""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS campaign_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        to_email TEXT NOT NULL,
        subject TEXT NOT NULL,
        content TEXT NOT NULL,
        delivery_status TEXT NOT NULL,
        idea_reference TEXT,
        sent_at TEXT NOT NULL
    )
    """)

    conn.commit()
    conn.close()
    logger.info(f"Database initialized and 'campaign_logs' table ensured in {DB_FILE}")

async def log_campaign_email(to_email: str, subject: str, content: str, delivery_status: str, idea_reference: str = None):
    """Asynchronously log an email campaign entry into the database."""
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

# Initialize DB on import or call this explicitly before running campaigns
init_db()
