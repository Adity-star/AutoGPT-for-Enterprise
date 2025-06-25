def get_idea_generation_prompt(posts):
    return f"""
You are a business advisor specializing in identifying high-potential, revenue-generating business ideas from user discussions. 
Based on the following Reddit post titles and contents, generate up to 5 innovative business ideas that have strong commercial viability.

For each idea, provide:
- A concise description of the business idea or product opportunity
- The type of insight: one of ["Pain Point", "Tool Idea", "Marketing Insight", "Question", "Trend"]
- Scores from 1 to 10 (where 10 is highest) for market demand, novelty, feasibility, and monetization potential
- The primary target audience (e.g., remote workers, parents, SaaS companies)
- Your confidence score (0-100) indicating how strong this opportunity is

**Important: Only include ideas where monetization potential is 7 or higher and confidence is 70 or above.**

Respond ONLY in strict JSON format as follows:
{{
  "ideas": [
    {{
      "idea": "...",
      "description": "...",
      "type": "Tool Idea",
      "market_demand": 8,
      "novelty": 7,
      "feasibility": 9,
      "monetization_potential": 9,
      "target_audience": "...",
      "confidence": 85
    }},
    ...
  ]
}}

Posts:
{posts}
""" 

