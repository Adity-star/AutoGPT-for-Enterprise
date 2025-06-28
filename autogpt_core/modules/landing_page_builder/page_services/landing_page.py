from pydantic import BaseModel
from typing import List

class LandingPageContent(BaseModel):
    headline: str
    subheadline: str
    features: List[str]
    call_to_action: str

    