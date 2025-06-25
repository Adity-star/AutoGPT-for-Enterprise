import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from utils.logger import logging


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(env_file=".env", extra='ignore', env_file_encoding='utf-8')

    # General Settings
    app_name: str = "AutoGPT For Enterprises"
    REDDIT_CLIENT_ID: str
    REDDIT_CLIENT_SECRET: str
    REDDIT_USER_AGENT: str

    GOOGLE_API_KEY: str
    SERPAPI_API_KEY: str

    def __init__(self, **kwargs):
        logging.info("Loading application settings from environment variables")
        super().__init__(**kwargs)  # Proper way to call parent class constructor
        # Ensure SHORT_TERM_MEMORY directory exists if it's configured

# Initialize settings
settings = Settings()
logging.info("Settings loaded successfully")
