from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_FAMILY_DIR = REPO_ROOT / "prompts" / "gepa-full-press"
PROMPT_FAMILY_ENVIRONMENTS = (
    "tool_accuracy",
    "target_execution",
    "supported_target",
    "cooperative_press",
    "full_press",
)


def resolve_prompt_family_dir(prompt_family_dir: str | Path | None = None) -> Path:
    if prompt_family_dir is None:
        return DEFAULT_PROMPT_FAMILY_DIR
    return Path(prompt_family_dir).expanduser().resolve()


def load_prompt_family_blocks(prompt_family_dir: str | Path | None = None) -> dict[str, str]:
    root = resolve_prompt_family_dir(prompt_family_dir)
    blocks: dict[str, str] = {}
    for environment_kind in PROMPT_FAMILY_ENVIRONMENTS:
        prompt_path = root / f"{environment_kind}.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Missing prompt for environment {environment_kind!r}: {prompt_path}"
            )
        blocks[environment_kind] = prompt_path.read_text().strip()
    return blocks


def prompt_path_for_environment(
    environment_kind: str,
    prompt_family_dir: str | Path | None = None,
) -> Path:
    return resolve_prompt_family_dir(prompt_family_dir) / f"{environment_kind}.txt"
