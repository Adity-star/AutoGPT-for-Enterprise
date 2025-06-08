import re
import json
from typing import TypedDict, List, Dict, Any


def safe_parse_ideas(response_text: str) -> List[Dict[str, str]]:
    try:
        # Try parsing entire response
        return json.loads(response_text).get("ideas", [])
    except json.JSONDecodeError:
        # Try extracting just the JSON part using regex
        match = re.search(r'\{.*"ideas"\s*:\s*\[.*\]\s*\}', response_text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0)).get("ideas", [])
            except json.JSONDecodeError:
                pass
        # Still failed
        print("❌ Could not parse response as JSON.")
        print("🔎 Raw model response:\n", response_text)
        return []
