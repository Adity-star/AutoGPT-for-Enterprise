 # HTML layout logic

import aiofiles
import os
from jinja2 import Environment, FileSystemLoader,select_autoescape
from .landing_page import LandingPageContent


# Set up Jinja2 environment
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "../templates")
env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=select_autoescape(["html", "j2"])
)


async def building_landing_page(content: LandingPageContent,
                                image_url: str,
                                theme: str = "minimal",
                                output_file: str = "landing_pafe.html",
                                return_as_string: bool = False,
                                include_meta: bool = True) -> str:
    """
    Build and optionally save a themed landing page HTML.
    
    Args:
        content: LandingPageContent (Pydantic model)
        image_url: str - image to embed
        theme: str - one of available template names (without .j2)
        output_file: str - where to save the HTML
        return_as_string: bool - if True, returns string instead of saving
        include_meta: bool - if True, adds meta description and og:image

    Returns:
        HTML string or writes to file depending on `return_as_string`
    """

    try: 
        template = env.get_template(f"{theme}.html.j2")
    except Exception as e:
        raise ValueError(f"Theme '{theme}' not found in templates directory: {e}")

    html = template.render(
        headline=content.headline,
        subheadline=content.subheadline,
        features=content.features,
        call_to_action=content.call_to_action,
        image_url=image_url,
        include_meta=include_meta
    )

    if return_as_string:
        return html

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    async with aiofiles.open(output_file, "w") as f:
        await f.write(html)

    return output_file
