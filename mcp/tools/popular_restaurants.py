"""yamyam-ops API를 호출해 인기 맛집을 조회하는 MCP 도구."""

import json
import os
import urllib.error
import urllib.parse
import urllib.request


def _fetch_json(url: str, timeout: int = 15) -> dict | list:
    """GET 요청으로 JSON 응답을 반환합니다. 오류 시 예외를 발생시킵니다."""
    req = urllib.request.Request(url, method="GET")
    req.add_header("Accept", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def get_popular_restaurants(
    limit: int | None = None,
    min_review_count: int | None = None,
    category_large: str | list[str] | None = None,
) -> str:
    """
    한식·중식 등 카테고리별 인기 맛집을 N곳 조회합니다. 예: "한식 인기 맛집 5곳"
    → limit=5, category_large="한식" 또는 ["한식"].

    yamyam-ops API(/kakao/diners/filtered)를 호출해 인기도(bayesian_score) 순으로
    맛집 목록을 반환합니다.
    YAMYAM_OPS_API_URL 환경 변수(예: http://localhost:8000)가 설정되어 있어야 합니다.

    Args:
        limit: 반환할 맛집 개수 (1~20). 미지정 시 10.
        min_review_count: 최소 리뷰 개수 이상만 조회 (선택).
        category_large: 대분류 카테고리. 한 개면 문자열("한식"), 여러 개면 리스트(["한식", "중식"])
        (선택).

    Returns:
        인기 맛집 목록을 요약한 문자열. API 오류 시 오류 메시지.
    """
    base_url = (os.environ.get("YAMYAM_OPS_API_URL") or "").rstrip("/")
    if not base_url:
        return (
            "오류: YAMYAM_OPS_API_URL 환경 변수가 설정되지 않았습니다. "
            ".env에 YAMYAM_OPS_API_URL를 설정하고, MCP 서버를 재시작하세요."
        )

    # limit이 None이어도 기본 10 적용 (LLM이 생략/null 보낼 수 있음)
    limit = 10 if limit is None else limit
    limit = max(1, min(20, limit))

    # LLM이 "한식" 하나만 넘길 수 있음 → 리스트로 통일
    if isinstance(category_large, str):
        categories = [category_large.strip()] if category_large.strip() else []
    else:
        categories = [c.strip() for c in (category_large or []) if c and str(c).strip()]
    query_parts = [f"limit={limit}"]
    if min_review_count is not None and min_review_count >= 0:
        query_parts.append(f"min_review_count={min_review_count}")
    for c in categories:
        query_parts.append(f"diner_category_large={urllib.parse.quote(c, safe='')}")

    filtered_url = f"{base_url}/kakao/diners/filtered?{'&'.join(query_parts)}"

    try:
        data = _fetch_json(filtered_url)
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        return f"API 오류 (HTTP {e.code}): {body or e.reason}"
    except urllib.error.URLError as e:
        return f"연결 오류: {e.reason}. YAMYAM_OPS_API_URL과 서버 상태를 확인하세요."
    except json.JSONDecodeError as e:
        return f"응답 파싱 오류: {e}"
    except Exception as e:
        return f"오류: {e}"

    if not isinstance(data, list):
        return f"예상하지 못한 응답 형식: {type(data)}"

    if not data:
        return "조건에 맞는 인기 맛집이 없습니다."

    # FilteredDinerResponse는 id, diner_idx, distance 만 반환 → 각 id로 상세 조회
    details: list[dict] = []
    for item in data[:limit]:
        diner_id = item.get("diner_idx")
        if not diner_id:
            continue
        detail_url = f"{base_url}/kakao/diners/{diner_id}"
        try:
            detail = _fetch_json(detail_url)
            if isinstance(detail, dict):
                details.append(detail)
        except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError):
            # 상세 조회 실패 시 id만으로 한 줄 추가
            details.append({"diner_idx": diner_id, "diner_name": "(상세 조회 실패)"})

    lines = [f"인기 맛집 {len(details)}곳 (인기도 순):"]
    for i, d in enumerate(details, 1):
        name = d.get("diner_name") or "-"
        category = d.get("diner_category_large") or ""
        addr = d.get("diner_road_address") or d.get("diner_num_address") or ""
        score = d.get("bayesian_score")
        rating = d.get("diner_review_avg")
        review_count = d.get("diner_review_cnt")
        parts = [f"{i}. {name}"]
        if category:
            parts.append(f" [{category}]")
        if addr:
            parts.append(f" - {addr}")
        if score is not None:
            parts.append(f" (인기도: {score:.2f})")
        if rating is not None:
            parts.append(f" 평점 {rating}")
        if review_count is not None:
            parts.append(f" 리뷰 {review_count}개")
        lines.append("".join(parts))

    return "\n".join(lines)
