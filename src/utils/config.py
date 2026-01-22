"""설정 로더 유틸리티."""

import asyncio
from pathlib import Path
from typing import Any

import yaml


def _get_model_config_sync(config_name: str | None = None) -> tuple[dict[str, Any], str]:
    """
    YAML 파일에서 모델 설정을 로드합니다.

    Args:
        config_name: 설정 파일 이름 (확장자 제외). 예: "gpt-4o-mini", "gemini-2.5-flash"
                    None이면 config/models/ 디렉토리에서 첫 번째 .yaml 파일을 사용합니다.

    Returns:
        (모델 설정 딕셔너리, 모델 이름) 튜플
        모델 이름은 파일명에서 추출됩니다 (확장자 제외).
    """
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent
    models_dir = project_root / "config" / "models"

    # 설정 파일 이름이 지정되지 않으면 첫 번째 .yaml 파일 찾기
    if config_name is None:
        yaml_files = list(models_dir.glob("*.yaml"))
        if not yaml_files:
            raise RuntimeError(
                f"설정 파일을 찾을 수 없습니다. {models_dir} 디렉토리에 .yaml 파일을 추가하세요."
            )
        # 첫 번째 파일 사용
        yaml_file = yaml_files[0]
        # 파일명에서 모델 이름 추출 (확장자 제외)
        model_name = yaml_file.stem
    else:
        # 설정 파일 경로: config/models/{config_name}.yaml
        yaml_file = models_dir / f"{config_name}.yaml"
        model_name = config_name

    if not yaml_file.exists():
        raise RuntimeError(f"설정 파일을 찾을 수 없습니다: {yaml_file}")

    with open(yaml_file, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
        return data, model_name


async def get_model_config(config_name: str | None = None) -> tuple[dict[str, Any], str]:
    """
    YAML 파일에서 모델 설정을 비동기적으로 로드합니다.

    Args:
        config_name: 설정 파일 이름 (확장자 제외). 예: "gpt-4o-mini", "gemini-2.5-flash"
                    None이면 config/models/ 디렉토리에서 첫 번째 .yaml 파일을 사용합니다.

    Returns:
        (모델 설정 딕셔너리, 모델 이름) 튜플
        모델 이름은 파일명에서 추출됩니다 (확장자 제외).
    """
    return await asyncio.to_thread(_get_model_config_sync, config_name)
