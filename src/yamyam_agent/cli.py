"""간단한 CLI 인터페이스."""

import sys

from .agent import YamyamAgent


def main() -> None:
    """CLI 메인 함수."""
    import argparse

    parser = argparse.ArgumentParser(description="Yamyam MCP Agent CLI")
    parser.add_argument(
        "--api-key",
        type=str,
        help="Gemini API 키 (없으면 환경 변수 GEMINI_API_KEY 사용)",
    )
    parser.add_argument(
        "--mcp-command",
        type=str,
        default="uv",
        help="MCP 서버 실행 명령어 (기본값: uv)",
    )
    parser.add_argument(
        "--mcp-args",
        nargs="+",
        default=["run", "python", "server.py"],
        help="MCP 서버 실행 인자 (기본값: run python server.py)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="사용할 Gemini 모델 (기본값: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="실행할 쿼리 (없으면 대화형 모드)",
    )

    args = parser.parse_args()

    try:
        # Agent 생성
        agent = YamyamAgent(
            gemini_api_key=args.api_key,
            mcp_command=args.mcp_command,
            mcp_args=args.mcp_args,
            model_name=args.model,
        )

        if args.query:
            # 단일 쿼리 실행
            print(f"\n질문: {args.query}")
            print("\n답변:")
            response = agent.run(args.query)
            print(response)
        else:
            # 대화형 모드
            print("Yamyam MCP Agent에 오신 것을 환영합니다!")
            print("'quit' 또는 'exit'를 입력하면 종료됩니다.\n")

            while True:
                try:
                    query = input("\n질문: ").strip()
                    if not query:
                        continue
                    if query.lower() in ["quit", "exit", "종료"]:
                        print("\n안녕히 가세요!")
                        break

                    print("\n답변:")
                    response = agent.run(query)
                    print(response)

                except KeyboardInterrupt:
                    print("\n\n안녕히 가세요!")
                    break
                except Exception as e:
                    print(f"\n에러 발생: {e}")

        # 리소스 정리
        agent.close()

    except Exception as e:
        print(f"에러: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
