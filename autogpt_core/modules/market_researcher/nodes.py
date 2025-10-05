
import os
import json
import time
import asyncio
import re
from typing import List, Dict, Any, Optional, Callable, Coroutine
from functools import wraps
from datetime import datetime, timedelta
from autogpt_core.core.secrets import secrets
from pytrends.request import TrendReq


from autogpt_core.modules.market_researcher.state import AnalysisConfig, MarketResearchState
from autogpt_core.modules.market_researcher.services.support_tools import safe_parse_json_from_llm, safe_parse_json_response
from autogpt_core.modules.market_researcher.services.support_tools import search_competitors
from modules.market_researcher.services.reddit_service import RedditService
from autogpt_core.utils.idea_memory import save_idea_to_db, save_trending_posts_to_db, load_recent_trending_posts_from_db
from utils.logger import logger

from autogpt_core.core.llm_service import LLMService
from autogpt_core.core.prompt_manager import render_prompt


# Instantiate LLM service globally (or inject)
llm_service = LLMService(provider="groq")



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
            return None 
        return wrapper
    return decorator



class AsyncResilientAnalyzer:
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self._llm_service = None

    def get_llm_service(self) -> LLMService:
        if self._llm_service is None:
            self._llm_service = LLMService(provider=self.config.llm_provider)
        return self._llm_service

    async def generate_content(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        if self.config.enable_caching:
            cached = await prompt_cache.get(prompt)
            if cached:
                logger.debug("Cache hit for prompt")
                return cached

        llm_service = self.get_llm_service()
        response = await llm_service.chat(prompt)

        if self.config.enable_caching:
            await prompt_cache.set(prompt, response, self.config.cache_ttl_minutes)
        return response

    async def generate_ideas_from_posts(self, posts: str) -> str:
        prompt = render_prompt("idea_generation", posts=posts)
        return await self.generate_content(prompt)



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
                
                if len(idea_text) > 5: 
                    ideas.append({
                        'idea': idea_text,
                        'trend_score': 50 
                    })
        
        return ideas[:10]  
        
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

def validate_env_keys(required_keys: list[str]) -> list[str]:
    """Check if all required keys are set and non-empty in secrets."""
    missing = [key for key in required_keys if not getattr(secrets, key, None)]
    return missing


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
        missing_reddit_keys = validate_env_keys(required_keys)
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
            


async def generate_idea_list(state: MarketResearchState) -> MarketResearchState:
    """Generate or accept user-provided business ideas using centralized LLMService and prompt_manager"""
    try:
        logger.info("Processing idea generation...")

        # Validate Groq API key (via secrets)
        missing_keys = validate_env_keys(["GROQ_API_KEY"])
        if missing_keys:
            error_msg = f"Missing Groq API key(s): {', '.join(missing_keys)}. Cannot generate ideas."
            logger.error(error_msg)
            state.errors.append(error_msg)
            state.idea_list = []
            return state

        # If user provided idea, skip generation
        if getattr(state, "user_idea", None):
            logger.info("User idea detected, skipping generation.")
            state.idea_list = [{"idea": state.user_idea}]
            return state

        # No trending posts fallback
        if not state.trending_posts:
            logger.warning("No trending posts available; skipping idea generation.")
            state.idea_list = []
            return state

        # Extract top trending topics
        topics = [post.get('title', '') for post in state.trending_posts[:5]]
        topics_text = '\n'.join(f"- {topic}" for topic in topics if topic)

        # Load and render prompt via prompt_manager
        try:
            prompt = render_prompt("idea_generation", posts=topics_text)
            logger.debug(f"Prompt used:\n{prompt}")
        except Exception as e:
            logger.warning(f"Failed to load idea_generation prompt template: {e}")
            prompt = get_idea_generation_prompt_fallback(topics_text) 


        # Use the safe wrapper to get JSON data with retries
        ideas_data = await safe_parse_json_from_llm(llm_service, prompt)

        if not ideas_data:
            error_msg = "Failed to get valid JSON ideas from LLM after retries."
            logger.error(error_msg)
            state.errors.append(error_msg)
            state.idea_list = []
            return state
        
        ideas = ideas_data.get("ideas", [])

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
        logger.info(f"Generated {len(validated_ideas)} validated ideas.")
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
        missing_keys = validate_env_keys(["GROQ_API_KEY"])
        if missing_keys:
            logger.error(f"Missing Groq API key(s): {', '.join(missing_keys)}. Cannot analyze demand.")
            return {
                **idea,
                "demand_analysis": "No demand analysis available (missing API key)",
                "demand_score": "NA",
                "analysis_type": "demand"
            }

        # Render prompt using prompt_manager
        try:
            prompt = render_prompt("demand_analysis", idea=idea.get("idea", ""))
        except Exception as e:
            logger.warning(f"Failed to load demand_analysis prompt template: {e}")
            prompt = f"""
            You are a business analyst. Respond in JSON as specified.

            Analyze the market demand for this business idea: "{idea.get('idea', '')}"
            Return a JSON object like:
            {{
              "demand_analysis": "short summary",
              "demand_score": number (1-10)
            }}
            """

        # Use centralized LLM service and parse with retries
        result = await safe_parse_json_response(llm_service, prompt)

        if not result:
            logger.error("Failed to get valid JSON from LLM for demand analysis.")
            return {
                **idea,
                "demand_analysis": "No demand analysis available (parse error)",
                "demand_score": "NA",
                "analysis_type": "demand"
            }

        return {
            **idea,
            "demand_analysis": result.get("demand_analysis", "No summary provided"),
            "demand_score": result.get("demand_score", "NA"),
            "analysis_type": "demand"
        }

    except Exception as e:
        logger.error(f"Demand analysis failed: {e}")
        return {
            **idea,
            "demand_analysis": "No demand analysis available (exception)",
            "demand_score": "NA",
            "analysis_type": "demand"
        }


async def analyze_single_idea_competition(idea: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # Validate SerpAPI key
        missing_keys = validate_env_keys(["SERP_API_KEY"])
        if not missing_keys:
            logger.error("SERP_API_KEY environment variable is not set")
            return {
                **idea,
                "competition_analysis": "No competition analysis available (missing SerpAPI key)",
                "competition_score": "NA",
                "analysis_type": "competition"
            }

        logger.info(f"Searching competitors for idea: {idea.get('idea', '')}")

        # Call the search_competitors function
        competitors = search_competitors(idea.get("idea", ""), max_results=5)

        # Construct a summary string from the competitors list
        if competitors:
            competition_summary = (
                f"Found {len(competitors)} competitors: " +
                ", ".join(competitors)
            )
            competition_score = min(len(competitors), 10)  
        else:
            competition_summary = "No significant competitors found."
            competition_score = 1

        return {
            **idea,
            "competition_analysis": competition_summary,
            "competition_score": competition_score,
            "analysis_type": "competition"
        }

    except Exception as e:
        logger.exception(f"Competition analysis failed: {e}")
        return {
            **idea,
            "competition_analysis": "No competition analysis available (exception)",
            "competition_score": "NA",
            "analysis_type": "competition"
        }


async def analyze_single_idea_economics(idea: Dict[str, Any]) -> Dict[str, Any]:
    try:
        logger.info(f"Analyzing economics for idea: {idea.get('idea', 'Unknown')}")

        # Validate Groq API key
        missing_keys = validate_env_keys(["GROQ_API_KEY"])
        if missing_keys:
            logger.error(f"Missing required keys: {missing_keys}")
            return {
                **idea,
                "unit_economics": "No economics analysis available (missing API key)", 
                "economics_score": "NA",
                "analysis_type": "economics"
            }

        # Render the economics analysis prompt
        try:
            prompt = render_prompt("economic_analysis", input=idea.get("idea", ""))
        except Exception as e:
            logger.warning(f"Failed to load economic_analysis prompt: {e}")
            prompt = f"""
            You are a financial analyst. Analyze this business idea:
            "{idea.get('idea', '')}"

            Respond ONLY in JSON format:
            {{
              "unit_economics": "Summary of revenue, cost structure, margins, etc.",
              "economics_score": number  # 1-10, where 10 is high profitability
            }}
            """

        # Use safe JSON parse wrapper that calls the LLM
        llm = LLMService(provider="groq")
        result = await safe_parse_json_response(llm, prompt)

        if not result:
            logger.error("Failed to get valid economics JSON from LLM")
            return {
                **idea,
                "unit_economics": "No economics analysis available (parse error)",
                "economics_score": "NA",
                "analysis_type": "economics"
            }

        return {
            **idea,
            "unit_economics": result.get("unit_economics", "No economics provided"),
            "economics_score": result.get("economics_score", "NA"),
            "analysis_type": "economics"
        }

    except Exception as e:
        logger.exception(f"Economics analysis failed: {e}")
        return {
            **idea,
            "unit_economics": "No economics analysis available",
            "economics_score": "NA",
            "analysis_type": "economics"
        }



def analyze_ideas_with_trends(state: MarketResearchState) -> MarketResearchState:
    ideas = state.idea_list or []
    pytrends = TrendReq()
    trend_scores = []

    for idea in ideas:
        keyword = idea.get("idea", idea) if isinstance(idea, dict) else idea
        try:
            pytrends.build_payload([keyword], timeframe='today 12-m')
            data = pytrends.interest_over_time()
            score = round(data[keyword].mean(), 2) if not data.empty else 0.0
        except Exception as e:
            logger.error(f"Error fetching trends for {keyword}: {e}")
            score = 0.0

        # Save trend score into the idea itself
        if isinstance(idea, dict):
            idea["trend_score"] = score

        trend_scores.append({"idea": keyword, "trend_score": score})

    state.idea_trend_scores = trend_scores
    return state


async def parallel_analysis(state: MarketResearchState) -> MarketResearchState:
    """Run parallel analysis on all ideas"""
    try:
        logger.info("Starting parallel analysis...")
        
        if not state.idea_list:
            logger.warning("No ideas to analyze")
            state.parallel_analysis = {"demand": [], "competition": [], "economics": []}
            return state
        
        ideas = state.idea_list

        batch_size = min(state.config.batch_size, 3) 
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
                # Extract metrics or set defaults
                final_score = float(idea.get("final_score", 70))
                demand_score = float(idea.get("demand_score", 7))
                competitors_count = int(idea.get("competitors_count", 5))

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
                    recommendation = "Highly Recommended"
                elif validation_score >= 6.5:
                    recommendation = "Recommended with Caution"
                else:
                    recommendation = " Not Recommended"

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

        if best_idea:
            logger.info(f"Best idea selected: {best_idea.get('idea', 'N/A')[:60]}")

            # Optionally save best idea
            try:
                from utils.idea_memory import save_idea_to_db
                save_idea_to_db(best_idea)
                logger.info(" Best idea saved to memory.")
            except Exception as save_err:
                logger.warning(f" Failed to save idea to DB: {save_err}")

        return state

    except Exception as e:
        logger.error(f"Validation process failed: {e}")
        state.errors.append(f"Validation failed: {str(e)}")
        state.validated_ideas = []
        state.best_business_idea = {}
        return state




async def store_results_to_file(state: MarketResearchState) -> MarketResearchState:
    """Store analysis results to a JSON file and save the best idea."""

    try:
        logger.info("Storing results to file...")

        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)

        # Safely extract config
        config = getattr(state, "config", {})
        def get_config_value(key, default=None):
            if isinstance(config, dict):
                return config.get(key, default)
            elif config is not None:
                return getattr(config, key, default)
            return default

        # Best idea structure
        best_idea = state.best_business_idea or {}

        # Execution timing
        execution_seconds = 0.0
        if getattr(state, "start_time", None):
            execution_seconds = (datetime.now() - state.start_time).total_seconds()

        # Prepare final output
        output_data = {
            "analysis_metadata": {
                "timestamp": datetime.now().isoformat(),
                "execution_time_seconds": round(execution_seconds, 2),
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

        # Write to JSON file
        filename = f"market_research_output_{int(time.time())}.json"
        output_path = os.path.join("data", filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f" Results successfully written to: {output_path}")

        # Clear expired prompt cache if applicable
        if prompt_cache:
            await prompt_cache.clear_expired()

        # Redundantly save best idea to DB
        if best_idea.get("idea"):
            try:
                save_idea_to_db(best_idea)
                logger.info(" Best idea saved to DB.")
            except Exception as db_err:
                logger.warning(f" Could not save best idea to DB: {db_err}")

        return state

    except Exception as e:
        error_msg = f" Failed to store results: {e}"
        logger.error(error_msg)
        state.errors.append(error_msg)
        return state




