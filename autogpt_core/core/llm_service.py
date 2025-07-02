from typing import Literal, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
import asyncio
import logging
import tiktoken
import time
from core.llm_cache import get_cached_response, save_response

from autogpt_core.core.secrets import secrets

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self, 
                 provider: Literal["openai", "groq"] = "groq",
                 max_retries: int = 3,
                 retry_delay: float = 2.0):
        self.provider = provider.lower()
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        if self.provider == "groq":
            self.llm: Runnable = ChatGroq(
                api_key=secrets.GROQ_API_KEY,
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                temperature=0.2,
                streaming=True,
            )
        elif self.provider == "openai":
            self.llm: Runnable = ChatOpenAI(
                api_key=secrets.OPENAI_API_KEY,
                model="gpt-4o",
                temperature=0.2,
                streaming=True,
            )
        elif self.provider == "Google":
            self.ll: Runnable = ChatGoogleGenerativeAI(
                api_key = secrets.GOOGLE_API_KEY,
                model = "gemini-2.0-flash",
                temperature=0.2,
                streaming=True
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    async def chat(self, prompt: str) -> str:
        model_name = self.llm.model_name

        # Cache check
        cached = get_cached_response(prompt, model=model_name)
        if cached:
            logger.info(f"Cache hit for model: {model_name}")
            return cached

        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=prompt)
        ]

        for attempt in range(self.max_retries):
            try:
                logger.info(f"LLM call (attempt {attempt + 1})")
                response = await self.llm.ainvoke(messages)
                save_response(prompt, response.content, model=model_name)
                return response.content
            except Exception as e:
                logger.warning(f"LLM error: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise

    async def stream_chat(self, prompt: str):
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=prompt)
        ]
        try:
            async for chunk in self.llm.astream(messages):
                yield chunk.content
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise

    def sync_chat(self, prompt: str) -> str:
        model_name = self.llm.model_name

        cached = get_cached_response(prompt, model=model_name)
        if cached:
            logger.info(f"Cache hit (sync) for model: {model_name}")
            return cached

        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=prompt)
        ]
        for attempt in range(self.max_retries):
            try:
                response = self.llm.invoke(messages)
                save_response(prompt, response.content, model=model_name)
                return response.content
            except Exception as e:
                logger.warning(f"Sync LLM error: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise


    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        try:
            encoding = tiktoken.encoding_for_model(model or "gpt-4o")
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
