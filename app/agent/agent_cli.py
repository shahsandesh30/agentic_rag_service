import argparse, json
from app.agent.graph import run_agent

def main():
    p = argparse.ArgumentParser(description="Run agent over a question.")
    p.add_argument("question", help="User question")
    p.add_argument("--trace", action="store_true")
    args = p.parse_args()
    out = run_agent(args.question)
    if args.trace:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(out["final"], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
