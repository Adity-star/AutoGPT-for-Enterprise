from pydantic import BaseModel
from typing import List, Optional, Dict


class CampaignInput(BaseModel):
    target_audience: str
    product_name: Optional[str] = None
    custom_message: Optional[str] = None


class CampaignState(BaseModel):
    target_audience: Optional[str] = None
    idea: Optional[Dict] = None
    contacts: Optional[List[str]] = []
    email_content: Optional[str] = None
    email_payload: Optional[Dict] = None
    send_status: Optional[str] = None
    custom_message: Optional[str] = None
