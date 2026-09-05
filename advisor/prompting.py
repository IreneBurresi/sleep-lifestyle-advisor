"""Prompt templates, one directory per version under advisor/prompts/.

prompts/v1/system.md     instructions, plain text
prompts/v1/user.md.j2    the user message, a Jinja template rendered from an Assignment
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Self

from jinja2 import Environment, FileSystemLoader, StrictUndefined, Template

from advisor.profiling import Assignment

PROMPTS = Path(__file__).parent / "prompts"


def available_versions() -> list[str]:
    return sorted(p.name for p in PROMPTS.iterdir() if (p / "system.md").exists())


@dataclass(frozen=True)
class Prompt:
    version: str
    system: str
    _template: Template

    @classmethod
    def load(cls, version: str) -> Self:
        directory = PROMPTS / version
        if not directory.exists():
            raise FileNotFoundError(
                f"no prompt {version!r} in {PROMPTS}; available: {available_versions()}"
            )
        env = Environment(
            loader=FileSystemLoader(directory),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )
        env.filters["g"] = lambda x: f"{x:g}"
        return cls(
            version=version,
            system=(directory / "system.md").read_text(),
            _template=env.get_template("user.md.j2"),
        )

    def render(self, a: Assignment, snippets: list | None = None) -> str:
        return self._template.render(
            a=a, p=a.profile, u=a.user, values=a.user.measures(), snippets=snippets or []
        )
