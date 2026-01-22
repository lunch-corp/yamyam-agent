"""MCP 도구 정의."""

import random

from fastmcp import FastMCP


def register_tools(mcp: FastMCP) -> None:
    """MCP 서버에 도구를 등록합니다."""

    @mcp.tool()
    def echo(query: str) -> str:
        """
        입력된 텍스트를 그대로 반환하는 간단한 도구입니다.

        Args:
            query: 반환할 텍스트

        Returns:
            입력된 텍스트
        """

        return query

    @mcp.tool()
    def recommend_menu(
        request: str,
        candidates: list[str] | None = None,
        exclude: list[str] | None = None,
        seed: int | None = None,
    ) -> str:
        """
        아주 간단한 규칙 기반 "메뉴 추천" 도구입니다(한글 입력 지원).

        - request에 포함된 키워드(예: "매운", "국물", "가벼운")로 후보를 점수화해 1개를 추천합니다.
        - candidates 미지정 시 기본 후보 목록에서 추천합니다.
        - 동점이면 seed 기반으로 재현 가능하게 선택합니다.

        Args:
            request: 사용자의 요청(한글/영문 혼합 가능)
            candidates: 추천 후보 목록(미지정 시 기본 메뉴)
            exclude: 제외할 후보 목록
            seed: 동점일 때 재현 가능한 선택을 위한 시드

        Returns:
            추천 결과(추천 메뉴 + 간단한 이유)
        """

        default_candidates = [
            "김치찌개",
            "된장찌개",
            "순두부찌개",
            "부대찌개",
            "국밥",
            "설렁탕",
            "비빔밥",
            "불고기",
            "삼겹살",
            "치킨",
            "피자",
            "라면",
            "칼국수",
            "냉면",
            "초밥",
            "샐러드",
            "샌드위치",
        ]

        candidates = candidates or default_candidates
        exclude_set = set(exclude or [])
        pool = [c for c in candidates if c and c not in exclude_set]
        if not pool:
            return "추천할 후보가 없습니다. candidates/exclude를 확인해주세요."

        req = (request or "").strip()
        req_l = req.lower()

        keyword_groups: dict[str, list[str]] = {
            "spicy": ["매운", "얼큰", "칼칼", "매콤", "spicy", "hot"],
            "soup": ["국물", "뜨끈", "따뜻", "탕", "국", "찌개", "soup"],
            "noodle": ["면", "라면", "국수", "칼국수", "냉면", "noodle"],
            "rice": ["밥", "든든", "비빔밥", "rice"],
            "meat": ["고기", "삼겹", "불고기", "치킨", "meat"],
            "light": ["가볍", "라이트", "다이어트", "샐러드", "light", "diet"],
        }

        def has_any(group: list[str]) -> bool:
            return any(k and (k.lower() in req_l) for k in group)

        wants = {k: has_any(v) for k, v in keyword_groups.items()}

        def score(item: str) -> tuple[int, list[str]]:
            it = item.lower()
            s = 0
            reasons: list[str] = []

            # 요청에 메뉴명이 직접 포함되면 최우선
            if req and it in req_l:
                s += 100
                reasons.append("요청에 메뉴명이 포함됨")

            if wants["spicy"] and any(x in it for x in ["김치", "부대", "순두부", "라면"]):
                s += 6
                reasons.append("매운/얼큰 선호")
            if wants["soup"] and any(x in it for x in ["찌개", "탕", "국밥", "설렁탕", "국"]):
                s += 5
                reasons.append("국물/뜨끈 선호")
            if wants["noodle"] and any(x in it for x in ["면", "라면", "국수", "칼국수", "냉면"]):
                s += 4
                reasons.append("면 선호")
            if wants["rice"] and any(x in it for x in ["밥", "비빔밥", "국밥"]):
                s += 3
                reasons.append("밥/든든 선호")
            if wants["meat"] and any(x in it for x in ["삼겹", "불고기", "치킨"]):
                s += 3
                reasons.append("고기 선호")
            if wants["light"] and any(x in it for x in ["샐러드", "샌드위치"]):
                s += 4
                reasons.append("가벼운 식사 선호")

            return s, reasons

        scored = [(item, *score(item)) for item in pool]  # (item, score, reasons)
        best_score = max(s for _, s, _ in scored)
        best = [(item, reasons) for item, s, reasons in scored if s == best_score]

        rng = random.Random(seed)
        picked_item, picked_reasons = rng.choice(best)
        why = ", ".join(picked_reasons) if picked_reasons else "기본 추천(랜덤)"
        return f"추천: {picked_item}\n이유: {why}\n(score={best_score})"
