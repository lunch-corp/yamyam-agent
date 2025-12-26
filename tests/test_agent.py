"""간단한 테스트 스크립트."""

from dotenv import load_dotenv

from yamyam_agent.agent import YamyamAgent

# 환경 변수 로드
load_dotenv()


def test_agent() -> None:
    """Agent 기본 테스트."""
    print("Yamyam Agent 테스트 시작...\n")

    try:
        # Agent 생성
        agent = YamyamAgent(model_name="gemini-2.5-flash")

        # 테스트 쿼리
        test_queries = [
            "안녕하세요!",
            "echo 도구를 사용해서 'Hello, World!'를 출력해주세요.",
        ]

        for query in test_queries:
            print(f"질문: {query}")
            print("\n답변:")
            response = agent.run(query)
            print(response)
            print("\n" + "=" * 50 + "\n")

        # 리소스 정리
        agent.close()
        print("테스트 완료!")

    except Exception as e:
        print(f"테스트 실패: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_agent()
