from utils.logger import logger

logger.info("This is an info message")
logger.warning("This is a warning")
logger.error("This is an error")



'''
# Basic usage
reddit_service = RedditService()
posts = reddit_service.get_trending_posts(["AItools", "startups"], limit=10)

# Business-focused trending posts
business_posts = reddit_service.get_business_trending_posts(limit=10)

# Custom subreddits with different sorting
posts = reddit_service.get_trending_posts(
    subreddits=["entrepreneur", "business"], 
    limit=10, 
    sort_by="rising"
)

'''
from modules.market_researcher.rebbit_service import RebbitService
reddit_service = RebbitService()

# Get trending posts from multiple subreddits
posts = reddit_service.get_business_trending_posts(limit=10)

print(f"\n=== TOP {len(posts)} TRENDING BUSINESS POSTS ===\n")

for i, post in enumerate(posts, 1):
    print(f"{i}. {post['title']}")
    print(f"   Subreddit: r/{post['subreddit']}")
    print(f"   Score: {post['score']} | Comments: {post['num_comments']}")
    print(f"   Trending Score: {post['trending_score']:.2f}")
    print(f"   Age: {post['age_hours']:.1f} hours")
    print(f"   URL: {post['permalink']}")
    print("-" * 80)


'''
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables (make sure your GOOGLE_API_KEY is in the environment)
load_dotenv()

# Configure the API key (required for Gemini)

# Create a model instance
model = genai.GenerativeModel("gemini-2.0-flash")

# Define the prompt
prompt = "Generate a list of 5 business ideas based on current tech trends"

# Generate content
response = model.generate_content(prompt)

# Print the generated text
print(response.text)
'''

