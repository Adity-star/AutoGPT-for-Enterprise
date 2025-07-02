from typing import Optional, Dict, List
from pydantic import BaseModel

class BlogWriterAgentState(BaseModel):
    idea_data: Optional[Dict] = None               
    research_summary: Optional[str] = None         
    blog_title: Optional[str] = None              
    blog_draft: Optional[str] = None              
    seo_keywords: Optional[List[str]] = None      
    meta_description: Optional[str] = None         
    suggestions: Optional[List[str]] = None        
