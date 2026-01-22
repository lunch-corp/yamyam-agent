"""프롬프트 로더 유틸리티."""

from pathlib import Path

import yaml


def get_system_prompt() -> str:
    """
    YAML 파일에서 system prompt를 로드합니다.

    Returns:
        System prompt 템플릿 문자열
    """
    # 프로젝트 루트의 prompts/system.yaml 파일 경로
    current_file = Path(__file__)
    # src/yamyam_agent/utils/prompt_loader.py -> prompts/system.yaml
    # src/utils/prompt_loader.py -> (project root)
    project_root = current_file.parent.parent.parent
    yaml_file = project_root / "prompts" / "system.yaml"

    if not yaml_file.exists():
        raise FileNotFoundError(f"프롬프트 파일을 찾을 수 없습니다: {yaml_file}")

    with open(yaml_file, encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return data["prompt"].strip()


def get_user_prompt() -> str:
    """
    YAML 파일에서 user prompt를 로드합니다.

    Returns:
        User prompt 템플릿 문자열
    """
    # 프로젝트 루트의 prompts/user.yaml 파일 경로
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent
    yaml_file = project_root / "prompts" / "user.yaml"

    if not yaml_file.exists():
        raise FileNotFoundError(f"프롬프트 파일을 찾을 수 없습니다: {yaml_file}")

    with open(yaml_file, encoding="utf-8") as f:
        data = yaml.safe_load(f)
        return data["prompt"].strip()
