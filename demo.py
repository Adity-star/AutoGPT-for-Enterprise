
from autogpt_core.utils.logger import logger
# from serpapi import GoogleSearch

# Use the logger
logger.info("I am the Best")
logger.error("Error message here")

# print("GoogleSearch imported successfully!")

# # from autogpt_core.services.support_tools import search_competitors 
# from autogpt_core.modules.market_researcher.services.rebbit_service import RedditService, get_trending_posts
# # # Basic usage
# reddit_service = RedditService()
# posts = reddit_service.get_business_trending_posts()

# # # Business-focused trending posts
# business_posts = reddit_service.get_business_trending_posts(limit=10)

# # # Custom subreddits with different sorting
# posts = reddit_service.get_trending_posts(
#     subreddits=["smallbusiness", "business"], 
#     limit=10, 
#     sort_by="rising"
#  )

# # search_competitors("AI automation and workflow tools")
# get_trending_posts(subreddit="business")

# print(f"\n=== TOP {len(posts)} TRENDING BUSINESS POSTS ===\n")

# for i, post in enumerate(posts, 1):
#     print(f"{i}. {post['title']}")
#     print(f"   Subreddit: r/{post['subreddit']}")
#     print(f"   Score: {post['score']} | Comments: {post['num_comments']}")
#     print(f"   Trending Score: {post['trending_score']:.2f}")
#     print(f"   Age: {post['age_hours']:.1f} hours")
#     print(f"   URL: {post['permalink']}")
#     print("-" * 80)

# from autogpt_core.modules.market_researcher.graph import run_market_research_agent,get_or_generate_market_research_idea


# async def main():
#     """Main entry point for the market research agent"""
#     try:
#         results = await get_or_generate_market_research_idea()
#         print("Market Research Results:")
#         for key, value in results.items():
#             print(f"{key}: {value}")
#     except Exception as e:
#         logger.error(f"Error running market research agent: {e}")

# if __name__ == "__main__":
#          import asyncio
#          asyncio.run(main())

import asyncio
from autogpt_core.modules.landing_page_builder.landing_page_graph import landing_page_graph  

from utils.logger import  logging
logging.basicConfig(level=logging.INFO)

async def main():
    # Compile the graph
    graph = landing_page_graph()

    # Run the graph starting with an empty state
    final_state = await graph.invoke({})

    print("\n✅ Landing page generated successfully!")
    print("HTML file saved at:", final_state.get("output_path", "Unknown"))

if __name__ == "__main__":
    asyncio.run(main())

