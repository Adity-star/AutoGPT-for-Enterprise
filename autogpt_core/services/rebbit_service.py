# services/reddit_service.py
import praw
import os
import time
import logging
from dotenv import load_dotenv
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import yaml

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_subreddits(category: str = "business", config_path: str = "autogpt_core/config/subreddits.yaml") -> List[str]:
    """Load subreddit names from YAML config"""
    try:
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
        return data.get(category, [])
    except Exception as e:
        logger.warning(f"Failed to load subreddits config: {e}")
        return []


class RedditService:
    def __init__(self):
        self.reddit = self._get_reddit_client()
        self.rate_limit_delay = 2  # seconds between requests
        self.last_request_time = 0
        
    def _get_reddit_client(self):
        """Initialize Reddit client with error handling"""
        try:
            return praw.Reddit(
                client_id=os.getenv("REDDIT_CLIENT_ID"),
                client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
                user_agent=os.getenv("REDDIT_USER_AGENT")
            )
        except Exception as e:
            logger.error(f"Failed to initialize Reddit client: {e}")
            raise
    
    def _rate_limit(self):
        """Implement rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _calculate_trending_score(self, post) -> float:
        """Calculate trending score based on multiple factors"""
        try:
            # Get post age in hours
            post_time = datetime.fromtimestamp(post.created_utc)
            age_hours = (datetime.now() - post_time).total_seconds() / 3600
            
            # Avoid division by zero
            if age_hours < 1:
                age_hours = 1
            
            # Calculate score: (upvotes + comments) / age_hours^1.5
            # This gives higher weight to recent posts with high engagement
            trending_score = (post.score + post.num_comments) / (age_hours ** 1.5)
            
            return trending_score
        except Exception as e:
            logger.warning(f"Error calculating trending score: {e}")
            return 0
    
    def _filter_post(self, post) -> bool:
        """Filter posts based on quality criteria"""
        try:
            # Filter criteria
            min_upvotes = 5
            min_comments = 2
            max_age_hours = 72  # Only posts from last 3 days
            
            # Check age
            post_time = datetime.fromtimestamp(post.created_utc)
            age_hours = (datetime.now() - post_time).total_seconds() / 3600
            
            # Apply filters
            if post.score < min_upvotes:
                return False
            if post.num_comments < min_comments:
                return False
            if age_hours > max_age_hours:
                return False
            if post.stickied:  # Skip pinned posts
                return False
            
            return True
        except Exception as e:
            logger.warning(f"Error filtering post: {e}")
            return False
    
    def get_trending_posts(self, 
                          subreddits: List[str] = ["AItools"], 
                          limit: int = 10,
                          sort_by: str = "hot") -> List[Dict]:
        """
        Get trending posts from multiple subreddits with filtering and ranking
        
        Args:
            subreddits: List of subreddit names
            limit: Number of posts to return (max 10)
            sort_by: Sorting method ('hot', 'top', 'new', 'rising')
        
        Returns:
            List of dictionaries containing post data
        """
        if limit > 10:
            limit = 10
            logger.warning("Limit exceeded 10, setting to maximum allowed (10)")
        
        all_posts = []
        
        for subreddit_name in subreddits:
            try:
                self._rate_limit()  # Rate limiting
                
                logger.info(f"Fetching posts from r/{subreddit_name}")
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Get posts based on sort method
                if sort_by == "hot":
                    posts = subreddit.hot(limit=50)  # Get more to filter from
                elif sort_by == "top":
                    posts = subreddit.top(time_filter="day", limit=50)
                elif sort_by == "new":
                    posts = subreddit.new(limit=50)
                elif sort_by == "rising":
                    posts = subreddit.rising(limit=50)
                else:
                    posts = subreddit.hot(limit=50)
                
                # Process posts
                for post in posts:
                    try:
                        # Apply filters
                        if not self._filter_post(post):
                            continue
                        
                        # Calculate trending score
                        trending_score = self._calculate_trending_score(post)
                        
                        post_data = {
                            'title': post.title,
                            'subreddit': subreddit_name,
                            'score': post.score,
                            'upvote_ratio': post.upvote_ratio,
                            'num_comments': post.num_comments,
                            'created_utc': post.created_utc,
                            'url': post.url,
                            'permalink': f"https://reddit.com{post.permalink}",
                            'author': str(post.author) if post.author else '[deleted]',
                            'trending_score': trending_score,
                            'selftext': post.selftext[:200] + '...' if len(post.selftext) > 200 else post.selftext,
                            'age_hours': (datetime.now() - datetime.fromtimestamp(post.created_utc)).total_seconds() / 3600
                        }
                        
                        all_posts.append(post_data)
                        
                    except Exception as e:
                        logger.warning(f"Error processing post: {e}")
                        continue
                
            except Exception as e:
                logger.error(f"Error fetching from r/{subreddit_name}: {e}")
                continue
        
        # Sort by trending score and return top posts
        sorted_posts = sorted(all_posts, key=lambda x: x['trending_score'], reverse=True)
        
        logger.info(f"Successfully fetched and processed {len(sorted_posts)} posts")
        return sorted_posts[:limit]
    
    def get_business_trending_posts(self, limit: int = 10) -> List[Dict]:
        """Get trending posts from business-related subreddits"""
        business_subreddits = load_subreddits("business")
        return self.get_trending_posts(
            subreddits=business_subreddits,
            limit=limit,
            sort_by="hot"
        )


# Convenience function for backward compatibility
def get_trending_posts(subreddit="AItools", limit=10):
    """Legacy function - uses new RedditService"""
    service = RedditService()
    return service.get_trending_posts([subreddit], limit)

