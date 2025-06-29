import sqlite3
from datetime import datetime
from typing import List, Dict, Any
import json
from datetime import datetime, timedelta
from utils.logger import logger

DB_FILE = "data/short_term_memory.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
         CREATE TABLE IF NOT EXISTS trending_posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_data TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS ideas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            idea TEXT,
            trend_score REAL,
            demand_analysis TEXT,
            demand_score REAL,
            competition_analysis TEXT,
            competition_score REAL,
            unit_economics TEXT,
            economics_score REAL,
            final_score REAL,
            scoring_breakdown TEXT,
            validation_score REAL,
            recommendation TEXT,
            validation_summary TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_trending_posts_to_db(posts: List[Dict[str, any]]) -> None:
    import sqlite3
    from datetime import datetime

    try:
        conn = sqlite3.connect("short_term_memory.db")
        c = conn.cursor()

        # Ensure table is created
        c.execute("""
            CREATE TABLE IF NOT EXISTS trending_posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                score REAL,
                fetched_at TEXT
            )
        """)

        # Clear previous entries (optional, keeps data fresh)
        c.execute("DELETE FROM trending_posts")

        # Insert new posts
        for post in posts:
            c.execute("INSERT INTO trending_posts (title, score, fetched_at) VALUES (?, ?, ?)", (
                post.get("title", ""),
                post.get("score", 0),
                datetime.utcnow().isoformat()
            ))

        conn.commit()
        conn.close()

    except Exception as e:
        logger.error(f"Failed to save trending posts: {e}")



TRENDING_POSTS_CACHE_EXPIRY_MINUTES = 60  # Change as needed

def load_recent_trending_posts_from_db(limit: int = 10) -> List[Dict[str, Any]]:
    """Load recent trending posts from the database if within cache window"""
    try:
        conn = sqlite3.connect("short_term_memory.db")
        c = conn.cursor()

        cutoff = (datetime.utcnow() - timedelta(minutes=TRENDING_POSTS_CACHE_EXPIRY_MINUTES)).isoformat()
        c.execute("""
            SELECT title, score FROM trending_posts 
            WHERE fetched_at >= ?
            ORDER BY score DESC 
            LIMIT ?
        """, (cutoff, limit))

        rows = c.fetchall()
        conn.close()

        return [{"title": row[0], "score": row[1]} for row in rows]
    except Exception as e:
        logger.error(f"Failed to load trending posts from DB: {e}")
        return []



def save_idea_to_db(idea: dict):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Delete all previous ideas to keep only one
    c.execute("DELETE FROM ideas")
     # Make sure table exists (you can also do this once at startup)
    c.execute("""
        CREATE TABLE IF NOT EXISTS ideas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            idea TEXT NOT NULL,
            final_score REAL,
            demand_analysis TEXT,
            demand_score INTEGER,
            competition_analysis TEXT,
            competition_score INTEGER,
            unit_economics TEXT,
            economics_score INTEGER,
            analysis_type TEXT
        )
    """)
    c.execute("""
        INSERT INTO ideas (
            idea, trend_score, demand_analysis, demand_score,
            competition_analysis, competition_score, unit_economics, economics_score,
            final_score, scoring_breakdown, validation_score, recommendation,
            validation_summary, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        idea.get("idea"),
        idea.get("trend_score"),
        idea.get("demand_analysis"),
        idea.get("demand_score"),
        idea.get("competition_analysis"),
        idea.get("competition_score"),
        idea.get("unit_economics"),
        idea.get("economics_score"),
        idea.get("final_score"),
        idea.get("scoring_breakdown"),
        idea.get("validation_score"),
        idea.get("recommendation"),
        idea.get("validation_summary"),
        datetime.utcnow().isoformat()
    ))
    conn.commit()
    conn.close()

def load_ideas_from_db(limit=1):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM ideas ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    columns = [desc[0] for desc in c.description]
    conn.close()
    return [dict(zip(columns, row)) for row in rows] 