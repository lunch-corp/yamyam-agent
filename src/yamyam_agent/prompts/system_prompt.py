"""System prompt 로더."""

from pathlib import Path


def get_system_prompt() -> str:
    """
    System prompt를 로드합니다.

    Returns:
        System prompt 템플릿 문자열
    """
    # 현재 파일의 디렉토리
    current_dir = Path(__file__).parent
    prompt_file = current_dir / "system_prompt.txt"

    if not prompt_file.exists():
        raise FileNotFoundError(f"프롬프트 파일을 찾을 수 없습니다: {prompt_file}")

    with open(prompt_file, encoding="utf-8") as f:
        return f.read().strip()
