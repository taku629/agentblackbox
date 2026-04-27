"""
動作確認用サンプル。実行するとDBにデータが入りダッシュボードで確認できる。
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentblackbox import BlackBox
import time


# デコレータパターン
@BlackBox.record(agent_name="researcher")
def research_agent(query: str) -> str:
    bb = BlackBox.current()
    # LLM呼び出しをシミュレート
    bb.record_llm_call(
        model="gpt-4o",
        input_text=f"Research this topic: {query}",
        output_text="Here are the key findings: AI agent adoption is growing 45% YoY...",
        input_tokens=150,
        output_tokens=320,
        latency_ms=1240,
    )
    # ツール呼び出しをシミュレート
    bb.record_tool_call(
        tool_name="web_search",
        args={"query": query, "num_results": 5},
        result=["result1", "result2", "result3"],
        latency_ms=340,
    )
    return "Research complete"


# コンテキストマネージャーパターン
def planner_agent(task: str):
    with BlackBox.session("planner") as bb:
        bb.record_llm_call(
            model="claude-3-5-sonnet-20241022",
            input_text=f"Plan this: {task}",
            output_text="Step 1: ... Step 2: ... Step 3: ...",
            input_tokens=80,
            output_tokens=200,
            latency_ms=890,
        )
        bb.record_tool_call(
            tool_name="calendar_check",
            args={"date": "2026-04-27"},
            result={"available": True},
            latency_ms=45,
        )


# エラーケース
@BlackBox.record(agent_name="buggy_agent")
def buggy_agent():
    bb = BlackBox.current()
    bb.record_llm_call(
        model="gpt-4o-mini",
        input_text="Do something",
        output_text="OK",
        input_tokens=10,
        output_tokens=5,
        latency_ms=200,
    )
    raise ValueError("Something went wrong in production!")


# マスキングのデモ
def masking_demo():
    BlackBox.configure(masking=True)
    with BlackBox.session("secure_agent") as bb:
        bb.record_llm_call(
            model="gpt-4o",
            input_text="User email: john@example.com, card: 4111-1111-1111-1111",
            output_text="Processed payment for john@example.com",
            input_tokens=30,
            output_tokens=20,
            latency_ms=500,
        )
    BlackBox.configure(masking=False)  # リセット


if __name__ == "__main__":
    print("Running examples...")
    research_agent("AI agent observability trends 2026")
    planner_agent("Launch AgentBlackBox on Product Hunt")
    try:
        buggy_agent()
    except ValueError:
        pass
    masking_demo()

    print("\nSessions recorded:")
    for s in BlackBox.list_sessions():
        print(f"  {s.session_id[:8]}  {s.agent_name:20s}  {s.status:8s}  ${s.total_cost_usd:.4f}")

    print("\nReplaying researcher session:")
    sessions = BlackBox.list_sessions(agent_name="researcher")
    if sessions:
        from agentblackbox.storage import SQLiteStorage, DEFAULT_DB_PATH
        session = SQLiteStorage(DEFAULT_DB_PATH).get_session(sessions[0].session_id)
        bb = BlackBox.__new__(BlackBox)
        bb._session = session
        bb._token = None
        bb.agent_name = session.agent_name
        bb.replay()
