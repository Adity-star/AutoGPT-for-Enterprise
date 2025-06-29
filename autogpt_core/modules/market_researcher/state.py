from dataclasses import dataclass, field
from datetime import datetime
from utils.logger import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

logging.info("Loading market research analysis configuration")

@dataclass
class AnalysisConfig:
    """Configuration for the market research analysis"""
    max_retries: int = 3
    batch_size: int = 5
    timeout: int = 30
    enable_caching: bool = True
    cache_ttl_minutes: int = 60
    model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    api_key: Optional[str] = None

    def validate(self):
        """Validate configuration parameters"""
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be non-negative: {self.max_retries}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1: {self.batch_size}")
        if self.timeout < 1:
            raise ValueError(f"timeout must be at least 1 second: {self.timeout}")
        if self.cache_ttl_minutes < 1:
            raise ValueError(f"cache_ttl_minutes must be at least 1 minute: {self.cache_ttl_minutes}")

@dataclass
class MarketResearchState:
    """State management for the market research workflow"""
    trending_posts: List[Dict[str, Any]] = field(default_factory=list)
    idea_list: List[Dict[str, Any]] = field(default_factory=list)
    parallel_analysis: Dict[str, List[Dict[str, Any]]] = field(
        default_factory=lambda: {"demand": [], "competition": [], "economics": []}
    )
    scored_ideas: List[Dict[str, Any]] = field(default_factory=list)
    validated_ideas: List[Dict[str, Any]] = field(default_factory=list)
    best_business_idea: Dict[str, Any] = field(default_factory=dict)
    config: AnalysisConfig = field(default_factory=AnalysisConfig)
    errors: List[str] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    user_idea: Optional[str] = None


class IdeaResponse(BaseModel):
    ideas: List[str]
