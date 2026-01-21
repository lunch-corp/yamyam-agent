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

### 모델/버전 관련 (404 NOT_FOUND 해결)

`models/gemini-1.5-flash is not found for API version v1beta` 같은 에러가 나면,
대부분 **모델 ID가 별칭(또는 폐기)** 이거나 **API 버전(v1beta)** 문제입니다.

- 기본값은 `gemini-1.5-flash-latest`를 사용합니다.
- 필요하면 아래 환경 변수를 설정하세요:

```bash
# (권장) v1로 강제 (일부 모델은 v1beta에서 404가 납니다)
export YAMYAM_GOOGLE_API_VERSION=v1

# 모델을 직접 지정 (예: ListModels 결과에서 generateContent 지원 모델로)
export YAMYAM_LLM_MODEL=gemini-1.5-flash-latest
```

## 사용법

### CLI 사용

#### MCP 서버를 별도로 띄우기 (추천)

```bash
# HTTP(SSE)로 MCP 서버 실행 (기본: http://127.0.0.1:8001/sse)
uv run python mcp/server.py --transport sse --host 127.0.0.1 --port 8001
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

## 요구사항

- Python 3.11 이상
- Gemini API 키


## 라이선스

LICENSE 파일을 참조하세요.
