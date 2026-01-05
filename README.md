# yamyam-agent

Yamyam MCP Agent - Gemini API와 LangChain을 사용한 MCP 기반 AI Agent

## 개요

이 프로젝트는 yamyam-mcp 서버와 통신하여 MCP 도구를 사용하는 Gemini 기반 AI Agent입니다. LangChain을 사용하여 구현되었으며, 간단한 CLI 인터페이스를 제공합니다.

## 설치

```bash
# 의존성 설치
uv sync

# 또는 pip 사용
pip install -e .
```

## 설정

`.env` 파일을 생성하고 Gemini API 키를 설정하세요:

```bash
cp .env.example .env
# .env 파일을 열어서 GEMINI_API_KEY를 설정
```

또는 환경 변수로 직접 설정:

```bash
export GEMINI_API_KEY=your_api_key_here
```

## 사용법

### CLI 사용

#### 대화형 모드

```bash
uv run yamyam-agent
```

#### 단일 쿼리 실행

```bash
uv run yamyam-agent --query "echo 도구를 사용해서 'Hello, World!'를 출력해주세요"
```

#### 옵션

- `--api-key`: Gemini API 키 (없으면 환경 변수 사용)
- `--mcp-command`: MCP 서버 실행 명령어 (기본값: uv)
- `--mcp-args`: MCP 서버 실행 인자 (기본값: run yamyam-mcp)
- `--model`: 사용할 Gemini 모델 (기본값: gemini-2.5-flash)
- `--query`: 실행할 쿼리 (없으면 대화형 모드)

### Python 코드에서 사용

```python
from yamyam_agent import YamyamAgent

# Agent 생성
agent = YamyamAgent(
    gemini_api_key="your_api_key",  # 또는 환경 변수 사용
    model_name="gemini-2.5-flash",  # 기본값
)

# 쿼리 실행
response = agent.run("echo 도구를 사용해서 'Hello'를 출력해주세요")
print(response)

# 리소스 정리
agent.close()
```

### 테스트

#### 프롬프트 로더 테스트 (간단)

```bash
# 프롬프트가 정상적으로 로드되는지 확인
uv run python test_prompt_loader.py
```

#### pytest를 사용한 단위 테스트

```bash
# 개발 의존성 설치
uv sync --extra dev

# 모든 테스트 실행
uv run pytest

# 프롬프트 관련 테스트만 실행
uv run pytest tests/test_prompts.py

# 통합 테스트 실행 (MCP 서버 연결 필요)
uv run pytest -m integration

# 상세 출력으로 테스트 실행
uv run pytest -v
```

#### 통합 테스트 스크립트

```bash
# Agent 전체 기능 테스트 (실제 MCP 서버 연결 필요)
uv run python test_agent.py
```

## 프로젝트 구조

```
yamyam-agent/
├── src/
│   └── yamyam_agent/
│       ├── __init__.py
│       ├── agent.py          # YamyamAgent 클래스
│       ├── mcp_client.py     # MCP 클라이언트 래퍼
│       ├── tool_adapter.py   # MCP 도구를 LangChain 도구로 변환
│       ├── cli.py            # CLI 인터페이스
│       └── prompts/          # 프롬프트 모듈
│           ├── __init__.py
│           ├── system_prompt.py
│           └── system_prompt.txt
├── tests/                    # pytest 테스트
│   ├── __init__.py
│   ├── test_prompts.py       # 프롬프트 테스트
│   └── test_agent_integration.py  # Agent 통합 테스트
├── test_agent.py             # 통합 테스트 스크립트
├── test_prompt_loader.py     # 프롬프트 로더 테스트 스크립트
├── pytest.ini                # pytest 설정
├── pyproject.toml
└── README.md
```

## 요구사항

- Python 3.11 이상
- yamyam-mcp 서버가 설치되어 있어야 함
- Gemini API 키

**참고**: yamyam-mcp 서버가 다른 디렉토리에 있는 경우, `--mcp-command`와 `--mcp-args` 옵션을 사용하여 경로를 지정할 수 있습니다. 예를 들어:

```bash
# yamyam-mcp가 상위 디렉토리에 있는 경우
cd /path/to/yamyam-mcp
uv run yamyam-mcp  # 이 명령이 PATH에 있어야 함

# 또는 절대 경로로 Python 스크립트 실행
uv run yamyam-agent --mcp-command python --mcp-args /path/to/yamyam-mcp/server.py
```

## 라이선스

LICENSE 파일을 참조하세요.
