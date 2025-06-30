
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
from autogpt_core.modules.landing_page_builder.landing_page_graph import generate_landing_page_images
from utils.logger import  logging
from autogpt_core.modules.landing_page_builder.landing_page_graph import generate_images
logging.basicConfig(level=logging.INFO)

async def main():
    # Compile the graph
    graph = landing_page_graph()

    # Run the graph starting with an empty state
    final_state = await graph.ainvoke({})

    print("\n✅ Landing page generated successfully!")
    print("HTML file saved at:", final_state.get("output_path", "Unknown"))

if __name__ == "__main__":
    asyncio.run(main())





# Assuming generate_images and generate_landing_page_images are imported already

# Setup logger if not already configured
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# import asyncio

# async def main():
#     # Mock landing page content data structure
#     class MockContent:
#         headline = "SuperWidget"
#         subheadline = "Making life easier"
#         features = ["Fast", "Reliable", "Easy to use"]

#     state = {
#         "content": MockContent()
#     }

#     # Run image generation
#     updated_state = await generate_images(state)

#     # Save image if available
#     image_bytes = updated_state.get("image_bytes")
#     if image_bytes:
#         image_path = "data/generated_image.png"
#         with open(image_path, "wb") as f:
#             f.write(image_bytes)
#         print(f"✅ Image saved to: {image_path}")
#     else:
#         print("❌ No image was generated.")

# if __name__ == "__main__":
#     asyncio.run(main())

# import asyncio
# import os
# import aiofiles
# from utils.logger import logger
# from autogpt_core.modules.landing_page_builder.landing_page_graph import building_landing_page
# from autogpt_core.modules.landing_page_builder.page_services.page_builder import building_landing_page
# from autogpt_core.modules.landing_page_builder.landing_page_graph import build_html_page

# import os
# import base64
# import asyncio
# import logging
# from autogpt_core.modules.landing_page_builder.page_services.landing_page import LandingPageContent

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

# async def main():
#     # Mock content matching your LandingPageContent model
#     class MockContent:
#         headline = "SuperWidget"
#         subheadline = "Making life easier"
#         features = ["Fast", "Reliable", "Easy to use"]
#         call_to_action = "Get Started Today!"

#     content = MockContent()

#     # Step 1: Generate image bytes async
#     logger.info("Starting image generation...")
#     image_result = await generate_landing_page_images(content)
    
#     if not image_result or "image_bytes" not in image_result:
#         logger.error("Image generation failed or returned no bytes.")
#         return
    
#     image_bytes = image_result["image_bytes"]
    
#     # Step 2: Convert image bytes to base64 data URL
#     image_base64 = base64.b64encode(image_bytes).decode("utf-8")
#     image_url = f"data:image/png;base64,{image_base64}"

#     # Step 3: Build and save the HTML landing page
#     logger.info("Building landing page HTML...")
#     output_path = await building_landing_page(
#         content=content,
#         image_url=image_url,
#         theme="minimal",
#         output_file="output/landing_page.html",
#         return_as_string=False
#     )
    
#     logger.info(f"Landing page saved at: {output_path}")

# if __name__ == "__main__":
#     asyncio.run(main())

