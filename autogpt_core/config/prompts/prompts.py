# core/prompt.py

from core.prompt_manager import render_prompt, get_prompt_metadata


def get_idea_generation_prompt(posts: str) -> str:
    return render_prompt("idea_generation", posts=posts)


def get_landing_page_prompt(
    idea: str,
    recommendation: str,
    demand_analysis: str,
    competition_analysis: str,
    unit_economics: str,
    format_instructions: str
) -> str:
    return render_prompt(
        "landing_page",
        idea=idea,
        recommendation=recommendation,
        demand_analysis=demand_analysis,
        competition_analysis=competition_analysis,
        unit_economics=unit_economics,
        format_instructions=format_instructions
    )


def get_email_generation_prompt(product: str, target_customer: str, benefits: str) -> tuple[str, str]:
    """
    Returns a tuple of (system_prompt, user_prompt)
    """
    system = get_prompt_metadata("email_outreach").get("system_prompt", "")
    prompt = render_prompt("email_outreach", product=product, target_customer=target_customer, benefits=benefits)
    return system, prompt
