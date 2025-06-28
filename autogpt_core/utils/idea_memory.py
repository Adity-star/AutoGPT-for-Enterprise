import sqlite3
from datetime import datetime

DB_FILE = "short_term_memory.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
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

def save_idea_to_db(idea: dict):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Delete all previous ideas to keep only one
    c.execute("DELETE FROM ideas")
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