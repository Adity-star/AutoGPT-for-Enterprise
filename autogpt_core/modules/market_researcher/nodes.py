from dotenv import load_dotenv
load_dotenv()

import os
import json
import time
import asyncio
import logging
import re
from typing import List, Dict, Any, Optional, Callable, Coroutine
from functools import wraps
from datetime import datetime, timedelta
import sys

import google.generativeai as genai
from .state import AnalysisConfig, MarketResearchState, IdeaResponse

from config.prompts.prompts import get_idea_generation_prompt
from .services.support_tools import analyze_ideas_with_trends, search_competitors
from .services.rebbit_service import RedditService
from settings import settings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from autogpt_core.utils.idea_memory import init_db, save_idea_to_db, save_trending_posts_to_db, load_recent_trending_posts_from_db
from utils.logger import logger
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)


class CacheEntry:
    """Cache entry with TTL support"""
    def __init__(self, value: str, ttl_minutes: int = 60):
        self.value = value
        self.created_at = datetime.now()
        self.ttl = timedelta(minutes=ttl_minutes)
    
    def is_expired(self) -> bool:
        return datetime.now() - self.created_at > self.ttl


class PromptCache:
    """Thread-safe prompt cache with TTL"""
    def __init__(self):
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[str]:
        async with self._lock:
            entry = self._cache.get(key)
            if entry and not entry.is_expired():
                return entry.value
            elif entry:  # Expired
                del self._cache[key]
            return None
    
    async def set(self, key: str, value: str, ttl_minutes: int = 60):
        async with self._lock:
            self._cache[key] = CacheEntry(value, ttl_minutes)
    
    async def clear_expired(self):
        """Clean up expired entries"""
        async with self._lock:
            expired_keys = [
                k for k, v in self._cache.items() if v.is_expired()
            ]
            for key in expired_keys:
                del self._cache[key]
            if expired_keys:
                logger.info(f"Cleared {len(expired_keys)} expired cache entries")


# Global cache instance
prompt_cache = PromptCache()


def async_retry(
    max_retries: int = 3, 
    backoff_base: float = 2.0, 
    backoff_factor: float = 0.1,
    retry_on_exceptions: tuple = (Exception,), 
    fail_fast_exceptions: tuple = (ValueError, KeyError)
):
    """Async retry decorator with exponential backoff"""
    def decorator(func: Callable[..., Coroutine]):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except fail_fast_exceptions as e:
                    logger.error(f"Fail-fast error on {func.__name__}: {e}")
                    raise
                except retry_on_exceptions as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Max retries reached for {func.__name__}: {e}")
                        raise
                    wait_time = (backoff_base ** attempt) + (backoff_factor * attempt)
                    logger.warning(f"Retry {attempt+1}/{max_retries} for {func.__name__}: {e}. Waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
            return None  # Should never reach here
        return wrapper
    return decorator


class AsyncResilientAnalyzer:
    """Async analyzer with caching and error handling"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self._llm = None

    def get_llm(self):
        if self._llm is None:
            self._llm = ChatGroq(
                model=self.config.model_name,  # e.g., "llama-3.3-70b-versatile"
                temperature=0.7,
                api_key=getattr(self.config, "api_key", None)
            )
        return self._llm

    @async_retry(max_retries=3, retry_on_exceptions=(Exception,))
    async def generate_content(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        if self.config.enable_caching:
            cached_result = await prompt_cache.get(prompt)
            if cached_result:
                logger.debug("Using cached result for prompt")
                return cached_result
            
        parser = JsonOutputParser(pydantic_object=IdeaResponse)
       
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", "Extract business ideas from the user input and return JSON."),
            ("user", "{input}")
        ])
        chain = chat_prompt | self.get_llm() | parser

        try:
            result: IdeaResponse = await chain.ainvoke({"input": prompt})
            text = json.dumps(result) 
            if self.config.enable_caching:
                await prompt_cache.set(prompt, text, self.config.cache_ttl_minutes)
            return text
        except Exception as e:
            logger.error(f"Content generation failed: {e}")
            raise


# Fallback implementations for missing services
def safe_parse_ideas(response_text: str) -> List[Dict[str, Any]]:
    """Parse ideas from LLM response with fallback"""
    try:
        # Try to extract JSON-like structures
        ideas = []
        lines = response_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and len(line) > 10:
                # Extract idea content
                idea_match = re.search(r'(?:idea|business|concept)[:.]?\s*(.+)', line, re.IGNORECASE)
                if idea_match:
                    idea_text = idea_match.group(1).strip(' ."')
                elif ':' in line:
                    idea_text = line.split(':', 1)[1].strip(' ."')
                else:
                    idea_text = line
                
                if len(idea_text) > 5:  # Valid idea length
                    ideas.append({
                        'idea': idea_text,
                        'trend_score': 50  # Default score
                    })
        
        return ideas[:10]  # Limit to top 10
        
    except Exception as e:
        logger.error(f"Idea parsing failed: {e}")
        return [{'idea': 'AI-powered productivity tools', 'trend_score': 70}]


def get_idea_generation_prompt_fallback(topics: str) -> str:
    """Fallback prompt generation"""
    return f"""
    Based on these trending topics:
    {topics}
    
    Generate 10 innovative business ideas. For each idea, provide:
    1. Clear business concept
    2. Target market
    3. Key value proposition
    
    Format each idea on a new line starting with "Idea: "
    """


def search_competitors_fallback(idea: str) -> List[Dict[str, Any]]:
    """Fallback competitor search"""
    return [
        {'title': f'Generic competitor for {idea[:20]}...', 'score': 50},
        {'title': f'Market leader in {idea[:15]}...', 'score': 80},
        {'title': f'Startup competitor for {idea[:18]}...', 'score': 30}
    ]


# Initialize global analyzer
analyzer = AsyncResilientAnalyzer(AnalysisConfig())


def get_missing_settings_keys(config_obj, keys: List[str]) -> List[str]:
    return [key for key in keys if not getattr(config_obj, key, "").strip()]

def get_default_trending_fallback() -> List[Dict[str, Any]]:
    return [
        {"title": "AI automation and workflow tools", "score": 95},
        {"title": "Sustainable packaging solutions", "score": 88},
        {"title": "Remote work productivity apps", "score": 85},
        {"title": "Health and wellness tracking", "score": 82},
        {"title": "Electric vehicle charging infrastructure", "score": 79},
        {"title": "Digital financial literacy platforms", "score": 76},
        {"title": "Local food delivery optimization", "score": 73},
        {"title": "Mental health support applications", "score": 70}
    ]



async def get_trending_industries(state: MarketResearchState) -> MarketResearchState:
    """Get trending posts: first from DB cache, then from Reddit API, then fallback."""
    try:
        logger.info("Fetching trending industries...")

        # Fetching from db if available always try cache lookup
        cached_posts = load_recent_trending_posts_from_db(limit=10)
        if cached_posts:
            logger.info(f"Loaded {len(cached_posts)} trending posts from DB cache.")
            state.trending_posts = cached_posts
            return state

        # If cache is empty, check keys and fetch from Reddit
        required_keys = ["REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT"]
        missing_reddit_keys = get_missing_settings_keys(settings, required_keys)
        if missing_reddit_keys:
            logger.warning(f"Missing Reddit API keys: {', '.join(missing_reddit_keys)}")
            fallback = get_default_trending_fallback()
            state.trending_posts = fallback
            state.errors.append(f"Missing Reddit API keys: {', '.join(missing_reddit_keys)} (used fallback data)")
            return state

        # Cache empty & credentials present—fetch from Reddit
        try:
            reddit = RedditService()
            top_posts = await asyncio.get_event_loop().run_in_executor(
                None, lambda: reddit.get_business_trending_posts()
            )
            if not top_posts:
                raise ValueError("Reddit returned no trending posts")
        except Exception as e:
            logger.warning(f"Reddit API failed: {e}")
            fallback = get_default_trending_fallback()
            state.trending_posts = fallback
            state.errors.append(f"Reddit API error; used fallback data.")
            return state

        # Save new posts to DB and return them
        save_trending_posts_to_db(top_posts)
        state.trending_posts = top_posts[:10]
        logger.info(f"Fetched and saved {len(state.trending_posts)} Reddit posts.")
        return state

    except Exception as e:
        error_msg = f"Failed to get trending industries: {e}"
        logger.error(error_msg)
        state.errors.append(error_msg)
        state.trending_posts = []
        return state


    except Exception as e:
        error_msg = f"Failed to get trending industries: {e}"
        logger.error(error_msg)
        state.errors.append(error_msg)
        state.trending_posts = []
        return state


       
async def generate_idea_list(state: MarketResearchState) -> MarketResearchState:
    """Generate or accept user-provided business ideas"""
    try:
        logger.info("Processing idea generation...")

        # Validate Groq API key
        required_keys = ["GROQ_API_KEY"]
        missing_reddit_keys = get_missing_settings_keys(settings, required_keys)
        if not required_keys:
            error_msg = "Missing Groq API key. Cannot generate ideas."
            logger.error(error_msg)
            state.errors.append(error_msg)
            state.idea_list = []
            return state

        if getattr(state, "user_idea", None):
            logger.info("User idea detected, Skipping generation.")
            state.idea_list = [{"idea": state.user_idea}]
            return state

        if not state.trending_posts:
            logger.warning("No trending posts available")
            state.idea_list = []
            return state
        
        # Extract topics
        topics = [post.get('title', '') for post in state.trending_posts[:5]]
        topics_text = '\n'.join(f"- {topic}" for topic in topics if topic)
        
        # Get prompt
        try:
            prompt = get_idea_generation_prompt(topics_text)
        except NameError:
            prompt = get_idea_generation_prompt_fallback(topics_text)
        
        # Generate ideas using Groq + LangChain
        response_text = await analyzer.generate_content(prompt)
        try:
            ideas_data = json.loads(response_text)
            ideas = ideas_data.get("ideas", [])
        except Exception as e:
            logger.error(f"Failed to parse ideas JSON: {e}")
            ideas = []
        
        # Validate and limit ideas
        validated_ideas = []
        for idea in ideas:
            if isinstance(idea, str) and len(idea) > 10:
                validated_ideas.append({"idea": idea})
            elif isinstance(idea, dict) and idea.get('idea') and len(idea['idea']) > 10:
                validated_ideas.append(idea)
            if len(validated_ideas) >= 5: 
                break
        
        state.idea_list = validated_ideas
        logger.info(f"Generated {len(validated_ideas)} validated ideas")
        return state
        
    except Exception as e:
        error_msg = f"Failed to generate ideas: {e}"
        logger.error(error_msg)
        state.errors.append(error_msg)
        state.idea_list = []
        return state


async def analyze_single_idea_demand(idea: Dict[str, Any]) -> Dict[str, Any]:

    logger.info(f"Analyzing demand for idea: {idea.get('idea', 'Unknown')}")
    try:
         # Validate Groq API key
        required_keys = ["GROQ_API_KEY"]
        missing_keys = get_missing_settings_keys(settings, required_keys)
        if not required_keys:
            logger.error("Missing Groq API key. Cannot analyze demand.")

            return {**idea,
                    "demand_analysis": "No demand analysis available (missing API key)", 
                    "demand_score": "NA",
                    "analysis_type": "demand"}
        
        # System message — no variables
        system_message = SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                template="You are a business analyst. Respond in JSON as specified.",
                input_variables=[]
            )
        )

        user_message = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template="""Analyze the market demand for this business idea: "{input}"
        Return a JSON object like:
        {{ 
        "demand_analysis": "short summary", 
        "demand_score": number (1-10) 
        }}""",
                input_variables=["input"]
            )
        )

        chat_prompt = ChatPromptTemplate(
            input_variables=["input"],
            messages=[system_message, user_message]
        )

        parser = JsonOutputParser(pydantic_object={
            "type": "object",
            "properties": {
                "demand_analysis": {"type": "string"},
                "demand_score": {"type": "number"}
            }
        })
        
        chain = chat_prompt | analyzer.get_llm() | parser
        result = await chain.ainvoke({"input": idea.get("idea", "")})
        return {**idea, **result, "analysis_type": "demand"}
    except Exception as e:
        logger.error(f"Demand analysis failed: {e}")
        return {**idea, "demand_analysis": "No demand analysis available", "demand_score": "NA", "analysis_type": "demand"}


async def analyze_single_idea_competition(idea: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # Validate Groq API key
        required_keys = ["GROQ_API_KEY"]
        missing_reddit_keys = get_missing_settings_keys(settings, required_keys)

        if not required_keys:
            logger.error("Missing Groq API key. Cannot analyze competition.")
            return {**idea,
                    "competition_analysis": "No competition analysis available (missing API key)",
                    "competition_score": "NA",
                    "analysis_type": "competition"}
        
        parser = JsonOutputParser(pydantic_object={
            "type": "object",
            "properties": {
                "competition_analysis": {"type": "string"},
                "competition_score": {"type": "number"}
            }
        })

         # Prompt Template
        chat_prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a business analyst. Respond in JSON as specified."),
                    ("user", """Analyze the competition for this business idea: "{input}"
        Return a JSON object like:
        {{ 
        "competition_analysis": "short summary", 
        "competition_score": number (1-10) 
        }}""")
                ])
        chain = chat_prompt | analyzer.get_llm() | parser
        result = await chain.ainvoke({"input": idea.get("idea", "")})
        return {**idea, **result, "analysis_type": "competition"}

    except Exception as e:
        logger.exception(f"Competition analysis failed: {e}")
        return {
            **idea,
            "competition_analysis": "No competition analysis available",
            "competition_score": "NA",
            "analysis_type": "competition"
        }

async def analyze_single_idea_economics(idea: Dict[str, Any]) -> Dict[str, Any]:
    try:
         # Validate Groq API key
        required_keys = ["GROQ_API_KEY"]
        missing_reddit_keys = get_missing_settings_keys(settings, required_keys)
        if not required_keys:
            logger.error("Missing Groq API key. Cannot analyze economics.")
            return {**idea, "unit_economics": "No economics analysis available (missing API key)", 
                    "economics_score": "NA",
                    "analysis_type": "economics"}
        
        # Output parser
        parser = JsonOutputParser(pydantic_object={
            "type": "object",
            "properties": {
                "unit_economics": {"type": "string"},
                "economics_score": {"type": "number"}
            }
        })

        # Prompt with escaped JSON
        chat_prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a business analyst. Respond in JSON as specified."),
                    ("user", """Estimate the unit economics for this business idea: "{input}"
        Return a JSON object like:
        {{ 
        "unit_economics": "short summary", 
        "economics_score": number (1-10) 
        }}""")
                ])
        
        chain = chat_prompt | analyzer.get_llm() | parser
        result = await chain.ainvoke({"input": idea.get("idea", "")})
        return {**idea, **result, "analysis_type": "economics"}

    except Exception as e:
        logger.exception(f"Economics analysis failed: {e}")
        return {
            **idea,
            "unit_economics": "No economics analysis available",
            "economics_score": "NA",
            "analysis_type": "economics"
        }

async def parallel_analysis(state: MarketResearchState) -> MarketResearchState:
    """Run parallel analysis on all ideas"""
    try:
        logger.info("Starting parallel analysis...")
        
        ideas = state.idea_list
        if not ideas:
            logger.warning("No ideas to analyze")
            state.parallel_analysis = {"demand": [], "competition": [], "economics": []}
            return state
        
        batch_size = min(state.config.batch_size, 3)  # Limit concurrency
        results = {"demand": [], "competition": [], "economics": []}
        
        # Analysis functions
        analysis_functions = {
            'demand': analyze_single_idea_demand,
            'competition': analyze_single_idea_competition,
            'economics': analyze_single_idea_economics
        }
        
        # Process ideas with controlled concurrency
        semaphore = asyncio.Semaphore(batch_size)
        
        async def analyze_idea_with_semaphore(idea: Dict[str, Any], analysis_type: str, func: Callable):
            async with semaphore:
                try:
                    result = await func(idea)
                    results[analysis_type].append(result)
                    logger.debug(f"Completed {analysis_type} analysis for: {idea.get('idea', 'unknown')[:30]}...")
                except Exception as e:
                    error_msg = f"{analysis_type.title()} analysis failed for '{idea.get('idea', 'unknown')}': {e}"
                    logger.error(error_msg)
                    state.errors.append(error_msg)
        
        # Create all analysis tasks
        tasks = []
        for idea in ideas:
            for analysis_type, func in analysis_functions.items():
                task = analyze_idea_with_semaphore(idea, analysis_type, func)
                tasks.append(task)
        
        # Execute all tasks
        await asyncio.gather(*tasks, return_exceptions=True)
        
        state.parallel_analysis = results
        logger.info(f"Parallel analysis completed - Demand: {len(results['demand'])}, "
                   f"Competition: {len(results['competition'])}, Economics: {len(results['economics'])}")
        
        return state
        
    except Exception as e:
        error_msg = f"Parallel analysis failed: {e}"
        logger.error(error_msg)
        state.errors.append(error_msg)
        return state


def extract_numerical_score(text: str, default: int = 5) -> int:
    """Extract numerical score from analysis text"""
    if not text or not isinstance(text, str):
        return default
    
    # Look for various score patterns
    patterns = [
        r'(\d{1,2})\s*(?:/\s*10|out\s+of\s+10)',  
        r'rating:\s*(\d{1,2})',                     
        r'score:\s*(\d{1,2})',                     
        r'(\d{1,2})/10',                           
        r'\b(\d{1,2})\s*(?:points?|pts?)\b'         
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                score = int(match.group(1))
                return max(1, min(score, 10))  
            except (ValueError, IndexError):
                continue
    
    return default


def combine_and_score(state: MarketResearchState) -> MarketResearchState:
    """Combine analyses and calculate final scores"""
    try:
        logger.info("Combining analyses and calculating scores...")
        
        # Group analyses by idea
        combined_ideas = {}
        
        for analysis_type in ["demand", "competition", "economics"]:
            for idea_data in state.parallel_analysis.get(analysis_type, []):
                idea_key = idea_data.get("idea", "")
                if not idea_key:
                    continue
                    
                if idea_key not in combined_ideas:
                    combined_ideas[idea_key] = idea_data.copy()
                else:
                    combined_ideas[idea_key].update(idea_data)
        
        if not combined_ideas:
            logger.warning("No combined ideas found")
            state.scored_ideas = []
            return state
        
        scored_ideas = []
        
        for idea_key, idea_data in combined_ideas.items():
            try:
                # Extract scores with defaults
                demand_score = extract_numerical_score(
                    idea_data.get("demand_analysis", ""), 6
                )
                competition_score = extract_numerical_score(
                    idea_data.get("competition_analysis", ""), 6
                )
                economics_score = extract_numerical_score(
                    idea_data.get("unit_economics", ""), 5
                )
                
                # Weighted scoring formula
                # Higher demand = better, Lower competition = better, Higher economics = better
                final_score = (
                    demand_score * 0.4 +           # 40% weight on demand
                    (11 - competition_score) * 0.3 + # 30% weight on low competition (inverted)
                    economics_score * 0.3          # 30% weight on economics
                ) * 10  # Scale to 100
                
                idea_data.update({
                    "demand_score": demand_score,
                    "competition_score": competition_score,
                    "economics_score": economics_score,
                    "final_score": round(final_score, 1),
                    "scoring_breakdown": f"Demand: {demand_score}/10, Competition: {competition_score}/10, Economics: {economics_score}/10"
                })
                
                scored_ideas.append(idea_data)
                
            except Exception as e:
                logger.error(f"Scoring failed for idea '{idea_key}': {e}")
                continue
        
        # Sort by final score (highest first)
        scored_ideas.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        
        state.scored_ideas = scored_ideas
        logger.info(f"Successfully scored {len(scored_ideas)} ideas")
        
        return state
        
    except Exception as e:
        error_msg = f"Scoring failed: {e}"
        logger.error(error_msg)
        state.errors.append(error_msg)
        state.scored_ideas = []
        return state


def validate_and_select(state: MarketResearchState) -> MarketResearchState:
    """Validate one or more ideas and select the best"""
    try:
        logger.info("Validating and selecting best ideas...")

        # Determine source of ideas
        scored_ideas = state.scored_ideas or state.idea_list

        if not scored_ideas:
            logger.warning("No ideas to validate")
            state.validated_ideas = []
            state.best_business_idea = {}
            return state

        # If only 1 idea (e.g. user-supplied), validate it alone
        top_ideas = scored_ideas[:3] if len(scored_ideas) > 1 else scored_ideas
        validated_ideas = []

        for idea in top_ideas:
            try:
                # Default values
                final_score = idea.get("final_score", 70)
                competitors_count = idea.get("competitors_count", 5)
                demand_score = idea.get("demand_score", 7)

                validation_score = 6.0

                # Adjust validation score based on criteria
                if final_score > 75:
                    validation_score += 2
                elif final_score > 60:
                    validation_score += 1

                if competitors_count < 3:
                    validation_score += 1.5
                elif competitors_count < 5:
                    validation_score += 0.5

                if demand_score >= 8:
                    validation_score += 1

                validation_score = min(validation_score, 10)

                # Recommendation
                if validation_score >= 8:
                    recommendation = "✅ Highly Recommended"
                elif validation_score >= 6.5:
                    recommendation = "🟡 Recommended with Caution"
                else:
                    recommendation = "❌ Not Recommended"

                idea.update({
                    "validation_score": round(validation_score, 1),
                    "recommendation": recommendation,
                    "validation_summary": f"Validation Score: {validation_score:.1f}/10 - {recommendation}"
                })

                validated_ideas.append(idea)

            except Exception as e:
                logger.error(f"Validation failed for idea: {e}")
                continue

        # Select best idea (or just the one)
        best_idea = max(validated_ideas, key=lambda x: x.get("final_score", 0)) if validated_ideas else {}

        state.validated_ideas = validated_ideas
        state.best_business_idea = best_idea

        logger.info(f"Validated {len(validated_ideas)} ideas, selected best: {best_idea.get('idea', 'None')[:50]}...")

        # Save to DB if it's a full idea
        if best_idea and best_idea.get("idea"):
            from utils.idea_memory import save_idea_to_db
            save_idea_to_db(best_idea)

        return state

    except Exception as e:
        logger.error(f"Validation process failed: {e}")
        state.errors.append(f"Validation failed: {str(e)}")
        state.validated_ideas = []
        state.best_business_idea = {}
        return state



async def store_results_to_file(state: MarketResearchState) -> MarketResearchState:
    """Store results to JSON file with proper serialization"""
    try:
        logger.info("Storing results to file...")
        
        # Create output directory
        os.makedirs("data", exist_ok=True)
        
        # Prepare serializable state
        if isinstance(state, dict):
            config = state.get("config", {})
        else:
            config = getattr(state, "config", None)
        def get_config_value(key):
            if isinstance(config, dict):
                return config.get(key)
            elif config is not None:
                return getattr(config, key, None)
            return None
        best_idea = state.best_business_idea or {}
        output_data = {
            "analysis_metadata": {
                "timestamp": datetime.now().isoformat(),
                "execution_time_seconds": (datetime.now() - state.start_time).total_seconds(),
                "total_ideas_processed": len(state.idea_list),
                "total_errors": len(state.errors)
            },
            "best_business_idea": {
                "idea": best_idea.get("idea", ""),
                "trend_score": best_idea.get("trend_score", 0),
                "demand_analysis": best_idea.get("demand_analysis", ""),
                "demand_score": best_idea.get("demand_score", ""),
                "competition_analysis": best_idea.get("competition_analysis", ""),
                "competition_score": best_idea.get("competition_score", ""),
                "unit_economics": best_idea.get("unit_economics", ""),
                "economics_score": best_idea.get("economics_score", ""),
                "final_score": best_idea.get("final_score", 0),
                "scoring_breakdown": best_idea.get("scoring_breakdown", ""),
                "validation_score": best_idea.get("validation_score", 0),
                "recommendation": best_idea.get("recommendation", ""),
                "validation_summary": best_idea.get("validation_summary", "")
            },
            "errors": state.errors,
            "config": {
                "max_retries": get_config_value("max_retries"),
                "batch_size": get_config_value("batch_size"),
                "timeout": get_config_value("timeout"),
                "model_name": get_config_value("model_name")
            }
        }
        
        # Write to file
        output_path = os.path.join("data", f"market_research_output_{int(time.time())}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results successfully stored at {output_path}")
        
        # Clean up expired cache entries
        await prompt_cache.clear_expired()
        
        # Save best idea to short-term memory DB (redundant but safe)
        if best_idea and best_idea.get("idea"):
            save_idea_to_db(best_idea)
        
        return state
        
    except Exception as e:
        error_msg = f"Failed to store results: {e}"
        logger.error(error_msg)
        state.errors.append(error_msg)
        return state



