# core/prompt_manager.py

import yaml
from jinja2 import Template
from pathlib import Path
from typing import Any, Dict

PROMPT_DIR = Path(__file__).parent.parent / "config" / "prompts"


class PromptTemplate:
    def __init__(self, name: str):
        self.name = name
        self.path = PROMPT_DIR / f"{name}.yaml"
        self.template: Template | None = None
        self.metadata: Dict[str, Any] = {}

        self._load()

    def _load(self):
        if not self.path.exists():
            raise FileNotFoundError(f"Prompt '{self.name}' not found at {self.path}")
        
        with open(self.path, "r", encoding="utf-8") as f:
            content = yaml.safe_load(f)

        if "template" not in content:
            raise ValueError(f"Prompt file {self.name}.yaml must contain a 'template' key.")

        self.template = Template(content["template"])
        self.metadata = content.get("metadata", {})

    def render(self, **kwargs) -> str:
        try:
            return self.template.render(**kwargs)
        except Exception as e:
            raise ValueError(f"Failed to render prompt '{self.name}': {e}")

    def get_metadata(self) -> Dict[str, Any]:
        return self.metadata


# Convenience helpers
def render_prompt(name: str, **kwargs) -> str:
    return PromptTemplate(name).render(**kwargs)


def get_prompt_metadata(name: str) -> Dict[str, Any]:
    return PromptTemplate(name).get_metadata()
