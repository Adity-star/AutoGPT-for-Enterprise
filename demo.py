import sys
print("Python executable:", sys.executable)



from autogpt_core.utils.logger import logger

from serpapi import GoogleSearch




# Use the logger
logger.info("I am the Best")
logger.error("Error message here")

print("GoogleSearch imported successfully!")

from autogpt_core.services.support_tools import search_competitors
#from autogpt_core.services.rebbit_service import RedditService

#from autogpt_core.services.rebbit_service import get_trending_posts
# # Basic usage
# reddit_service = RedditService()
# #posts = reddit_service.get_trending_posts(["AItools", "startups"], limit=10)

# # Business-focused trending posts
# business_posts = reddit_service.get_business_trending_posts(limit=10)

# # Custom subreddits with different sorting
# posts = reddit_service.get_trending_posts(
#     subreddits=["smallbusiness", "business"], 
#     limit=10, 
#     sort_by="rising"
#  )

search_competitors("AI automation and workflow tools")
#get_trending_posts(subreddit="business")

# from autogpt_core.modules.market_researcher.rebbit_service import RebbitService
# reddit_service = RebbitService()

# # Get trending posts from multiple subreddits
# posts = reddit_service.get_business_trending_posts(limit=10)

# print(f"\n=== TOP {len(posts)} TRENDING BUSINESS POSTS ===\n")

# for i, post in enumerate(posts, 1):
#     print(f"{i}. {post['title']}")
#     print(f"   Subreddit: r/{post['subreddit']}")
#     print(f"   Score: {post['score']} | Comments: {post['num_comments']}")
#     print(f"   Trending Score: {post['trending_score']:.2f}")
#     print(f"   Age: {post['age_hours']:.1f} hours")
#     print(f"   URL: {post['permalink']}")
#     print("-" * 80)




