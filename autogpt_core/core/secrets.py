import logging
from pydantic_settings import BaseSettings, SettingsConfigDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecretsManager(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore" 
    )

    # API keys and secrets
    REDDIT_CLIENT_ID: str
    REDDIT_CLIENT_SECRET: str
    REDDIT_USER_AGENT: str

    GOOGLE_API_KEY: str
    SERPAPI_API_KEY: str

    GROQ_API_KEY: str

    HF_TOKEN: str
    SENDGRID_API_KEY: str
    OPENAI_API_KEY: str

    SERP_API_KEY: str


    
    APP_NAME: str = "AutoGPT For Enterprises"

    def validate(self):
       
        missing_keys = []
        for key, value in self.dict().items():
            if value is None:
                missing_keys.append(key)
        if missing_keys:
            raise RuntimeError(f"Missing required environment variables: {missing_keys}")
        logger.info("All required environment variables are present.")

# Initialize settings once, reuse across your app
secrets = SecretsManager()
secrets.validate()
logger.info(f"Settings loaded successfully for app: {secrets.APP_NAME}")
