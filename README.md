# yamyam-agent

Yamyam MCP Agent - Gemini API와 LangChain을 사용한 MCP 기반 AI Agent

## 개요

이 프로젝트는 **레포 내부의 `mcp/server.py`(FastMCP)** 와 통신하여 MCP 도구를 사용하는 Gemini 기반 AI Agent입니다. LangChain을 사용하여 구현되었으며, 간단한 CLI 인터페이스를 제공합니다.

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

#### MCP 서버를 별도로 띄우기 (추천)

```bash
# HTTP(SSE)로 MCP 서버 실행 (기본: http://127.0.0.1:8001/sse)
uv run python mcp/server.py --transport sse --host 127.0.0.1 --port 8001
```

#### 대화형 모드

```bash
uv run yamyam-agent --mcp-url "http://127.0.0.1:8001/sse"
```

#### 단일 쿼리 실행

```bash
uv run yamyam-agent --mcp-url "http://127.0.0.1:8001/sse" --query "echo 도구를 사용해서 'Hello, World!'를 출력해주세요"
```

#### (옵션) CLI가 MCP 서버를 직접 띄우기 (stdio)

```bash
uv run yamyam-agent --spawn-mcp --query "안녕하세요!"
```

#### 옵션

- `--api-key`: Gemini API 키 (없으면 환경 변수 사용)
- `--mcp-url`: 이미 실행 중인 MCP 서버 URL (예: `http://127.0.0.1:8001/sse` 또는 `.../mcp`)
- `--mcp-url-transport`: 전송 방식 (sse | streamable-http). 미지정 시 URL로 추정
- `--spawn-mcp`: CLI가 MCP 서버를 직접 실행(stdio)
- `--mcp-command`: (spawn 시) MCP 서버 실행 명령어
- `--mcp-args`: (spawn 시) MCP 서버 실행 인자
- `--model`: 사용할 Gemini 모델 (기본값: gemini-2.5-flash)
- `--query`: 실행할 쿼리 (없으면 대화형 모드)

### Python 코드에서 사용

```python
from yamyam_agent import YamyamAgent

# Agent 생성
agent = YamyamAgent(
    # (참고) 현재 구현은 LLM 추론 대신 MCP 도구 정보를 보여주는 최소 기능입니다.
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

## Studio 연결 (LangGraph Studio)

LangGraph Studio의 **"Connect Studio to local agent"** 는 MCP 서버가 아니라,
`langgraph dev`로 띄운 **LangGraph 로컬 서버**에 연결합니다.

### 1) MCP 서버를 호스팅으로 띄우기

```bash
# MCP 서버를 SSE로 실행 (기본 경로: /sse)
uv run python mcp/server.py --transport sse --host 127.0.0.1 --port 8001
```

### 2) LangGraph 로컬 서버 실행 (Studio가 붙는 서버)

```bash
uv sync --extra studio

# (선택) 위에서 띄운 MCP 서버를 Agent가 쓰도록 설정
export YAMYAM_MCP_URL="http://127.0.0.1:8001/sse"
export YAMYAM_MCP_URL_TRANSPORT="sse"

# Studio 연결용 서버 실행 (Base URL = http://127.0.0.1:8000)
uv run langgraph dev --port 8000
```

이제 Studio에서 Base URL에 `http://127.0.0.1:8000`을 넣으면 연결됩니다.

## 프로젝트 구조

```
yamyam-agent/
├── mcp/
│   └── server.py              # FastMCP 기반 MCP 서버 (stdio)
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
- Gemini API 키

**참고**: 기본 MCP 서버(`mcp/server.py`) 대신 다른 MCP 서버를 쓰려면 `--mcp-command` / `--mcp-args` 옵션으로 실행 명령을 바꿀 수 있습니다. 예를 들어:

```bash
# 또는 절대 경로로 Python 스크립트 실행
uv run yamyam-agent --mcp-command python --mcp-args /path/to/any-mcp-server.py
```

## 라이선스

LICENSE 파일을 참조하세요.
