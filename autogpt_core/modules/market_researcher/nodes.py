import os
import json
import time
import asyncio
import logging
import re
from aiohttp import ClientTimeout
from typing import List, Dict, Any, Optional, Callable, Coroutine
from functools import wraps
from datetime import datetime, timedelta
import sys

import google.generativeai as genai
from langgraph.graph import StateGraph
from .state import AnalysisConfig, MarketResearchState

from config.prompts import get_idea_generation_prompt
from .services.support_tools import analyze_ideas_with_trends, search_competitors
from .services.rebbit_service import RedditService
from settings import Settings


# Configure structured logging
os.makedirs("logs", exist_ok=True)
log_file = os.path.join('logs', 'market_research.log')
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(message)s'))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)



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
        self._model = None
        self._session_timeout = ClientTimeout(total=config.timeout)
    
    def get_model(self) -> genai.GenerativeModel:
        """Get or create the model"""
        if self._model is None:
            try:
                self._model = genai.GenerativeModel(self.config.model_name)
                logger.info(f"Model {self.config.model_name} initialized")
            except Exception as e:
                logger.error(f"Failed to initialize model: {e}")
                raise
        return self._model
    
    @async_retry(max_retries=3, retry_on_exceptions=(Exception,))
    async def generate_content(self, prompt: str) -> str:
        """Generate content using Gemini API with caching"""
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        # Check cache first
        if self.config.enable_caching:
            cached_result = await prompt_cache.get(prompt)
            if cached_result:
                logger.debug("Using cached result for prompt")
                return cached_result
        
        # Generate new content
        model = self.get_model()
        try:
            # Run in thread executor since Gemini API is synchronous
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: model.generate_content(prompt)
            )
            
            if not response or not response.text:
                raise ValueError("Empty response from model")
            
            # Cache the result
            if self.config.enable_caching:
                await prompt_cache.set(
                    prompt, 
                    response.text, 
                    self.config.cache_ttl_minutes
                )
            
            return response.text.strip()
            
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


async def get_trending_industries(state: MarketResearchState) -> MarketResearchState:
    """Get trending posts from Reddit or use fallback data"""
    try:
        logger.info("Fetching trending industries...")

        if not all(Settings.REDDIT_CLIENT_ID,Settings.REDDIT_CLIENT_SECRET, Settings.REDDIT_USER_AGENT):
            error_msg = "Reddit API Credentials are missing or incomplete."
            logger.error(error_msg)
            state.errors.append(error_msg)
            return state

        # Try to use RedditService if available
        try:
            reddit = RedditService()
            
            # Run Reddit API call in thread executor
            top_posts = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: reddit.get_business_trending_posts()
            )
        except (NameError, AttributeError, Exception) as e:
            logger.warning(f"Reddit service unavailable: {e}. Using fallback data.")
            top_posts = None
        
        if not top_posts:
            logger.info("Using fallback trending topics")
            top_posts = [
                {"title": "AI automation and workflow tools", "score": 95},
                {"title": "Sustainable packaging solutions", "score": 88},
                {"title": "Remote work productivity apps", "score": 85},
                {"title": "Health and wellness tracking", "score": 82},
                {"title": "Electric vehicle charging infrastructure", "score": 79},
                {"title": "Digital financial literacy platforms", "score": 76},
                {"title": "Local food delivery optimization", "score": 73},
                {"title": "Mental health support applications", "score": 70}
            ]
        
        state.trending_posts = top_posts
        logger.info(f"Retrieved {len(top_posts)} trending posts")
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
        
        # Generate ideas
        response_text = await analyzer.generate_content(prompt)
        ideas = safe_parse_ideas(response_text)
        
        # Validate and limit ideas
        validated_ideas = []
        for idea in ideas:
            if isinstance(idea, dict) and idea.get('idea') and len(idea['idea']) > 10:
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
    """Analyze market demand for a single idea"""
    try:
        trend_score = idea.get('trend_score', 50)
        idea_text = idea.get('idea', 'Unknown idea')
        
        prompt = f"""
        Analyze the market demand for this business idea: "{idea_text}"
        
        Current trend score: {trend_score}/100
        
        Provide:
        1. Market demand rating (1-10 scale)
        2. Target audience size
        3. Market growth potential
        4. Key demand drivers
        
        Keep analysis concise (2-3 sentences) and include a numerical rating.
        """
        
        analysis_text = await analyzer.generate_content(prompt)
        
        return {
            **idea, 
            "demand_analysis": analysis_text, 
            "analysis_type": "demand"
        }
        
    except Exception as e:
        logger.error(f"Demand analysis failed for '{idea.get('idea', 'unknown')}': {e}")
        return {
            **idea, 
            "demand_analysis": f"Analysis unavailable: {str(e)[:100]}", 
            "analysis_type": "demand"
        }


async def analyze_single_idea_competition(idea: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze competition for a single idea"""
    try:
        idea_text = idea.get('idea', 'Unknown idea')
        
        # Get competitors
        try:
            competitors = search_competitors(idea_text)[:3]
        except NameError:
            competitors = search_competitors_fallback(idea_text)[:3]
        
        competitors_summary = []
        for comp in competitors:
            if isinstance(comp, dict):
                title = comp.get('title', str(comp))[:50]
            else:
                title = str(comp)[:50]
            competitors_summary.append(title)
        
        competitors_text = ', '.join(competitors_summary)
        
        prompt = f"""
        Analyze the competitive landscape for: "{idea_text}"
        
        Known competitors: {competitors_text}
        
        Provide:
        1. Competition intensity rating (1-10 scale, where 10 = very competitive)
        2. Market saturation level
        3. Differentiation opportunities
        4. Competitive advantages needed
        
        Keep analysis concise (2-3 sentences) and include a numerical rating.
        """
        
        analysis_text = await analyzer.generate_content(prompt)
        
        return {
            **idea, 
            "competition_analysis": analysis_text, 
            "competitors_count": len(competitors), 
            "analysis_type": "competition"
        }
        
    except Exception as e:
        logger.error(f"Competition analysis failed for '{idea.get('idea', 'unknown')}': {e}")
        return {
            **idea, 
            "competition_analysis": f"Analysis unavailable: {str(e)[:100]}", 
            "competitors_count": 5,  # Default assumption
            "analysis_type": "competition"
        }


async def analyze_single_idea_economics(idea: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze unit economics for a single idea"""
    try:
        idea_text = idea.get('idea', 'Unknown idea')
        
        prompt = f"""
        Estimate unit economics for this business: "{idea_text}"
        
        Provide realistic estimates for:
        1. Customer Acquisition Cost (CAC): $X
        2. Average Revenue Per Customer (ARPC): $X/month or $X/year
        3. Gross Margin: X%
        4. Break-even timeline: X months
        5. Economics viability rating: X/10
        
        Keep estimates realistic and provide brief justification (2-3 sentences).
        """
        
        economics_text = await analyzer.generate_content(prompt)
        
        return {
            **idea, 
            "unit_economics": economics_text, 
            "analysis_type": "economics"
        }
        
    except Exception as e:
        logger.error(f"Economics analysis failed for '{idea.get('idea', 'unknown')}': {e}")
        return {
            **idea, 
            "unit_economics": f"Analysis unavailable: {str(e)[:100]}", 
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
    """Validate ideas and select the best one"""
    try:
        logger.info("Validating and selecting best ideas...")
        
        scored_ideas = state.scored_ideas
        if not scored_ideas:
            logger.warning("No scored ideas to validate")
            state.validated_ideas = []
            state.best_business_idea = {}
            return state
        
        # Take top 3 ideas for validation
        top_ideas = scored_ideas[:3]
        validated_ideas = []
        
        for idea in top_ideas:
            try:
                # Base validation score
                validation_score = 6.0
                
                # Scoring criteria
                final_score = idea.get("final_score", 0)
                competitors_count = idea.get("competitors_count", 10)
                demand_score = idea.get("demand_score", 5)
                
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
                
                # Cap at 10
                validation_score = min(validation_score, 10)
                
                # Generate recommendation
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
        
        # Select the best idea (highest final score among validated)
        if validated_ideas:
            best_idea = max(validated_ideas, key=lambda x: x.get("final_score", 0))
        else:
            best_idea = {}
        
        state.validated_ideas = validated_ideas
        state.best_business_idea = best_idea
        
        logger.info(f"Validated {len(validated_ideas)} ideas, selected best: {best_idea.get('idea', 'None')[:50]}...")
        
        return state
        
    except Exception as e:
        error_msg = f"Validation failed: {e}"
        logger.error(error_msg)
        state.errors.append(error_msg)
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
        output_data = {
            "analysis_metadata": {
                "timestamp": datetime.now().isoformat(),
                "execution_time_seconds": (datetime.now() - state.start_time).total_seconds(),
                "total_ideas_processed": len(state.idea_list),
                "total_errors": len(state.errors)
            },
            "best_business_idea": state.best_business_idea,
            "errors": state.errors,
            "config": {
                "max_retries": state.config.max_retries,
                "batch_size": state.config.batch_size,
                "timeout": state.config.timeout,
                "model_name": state.config.model_name
            }
        }
        
        # Write to file
        output_path = os.path.join("data", f"market_research_output_{int(time.time())}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results successfully stored at {output_path}")
        
        # Clean up expired cache entries
        await prompt_cache.clear_expired()
        
        return state
        
    except Exception as e:
        error_msg = f"Failed to store results: {e}"
        logger.error(error_msg)
        state.errors.append(error_msg)
        return state



