"""
basic_usage.py — agentblackbox の全パターンを実際に動かすデモ
外部 API 不要。ローカル SQLite に記録されます。
"""
import time
import random
import json
import sys
import os

# パッケージをローカルから読み込む
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agentblackbox import BlackBox

# ============================================================
# パターン 1: デコレータ
# ============================================================
@BlackBox.record(agent_name="decorator_agent")
def run_decorator_agent(task: str) -> str:
    bb = BlackBox.current()
    assert bb is not None, "BlackBox.current() should return the active session"

    # LLM 呼び出しをシミュレート
    t0 = time.perf_counter()
    time.sleep(0.05)  # simulate network latency
    duration_ms = (time.perf_counter() - t0) * 1000

    bb.record_llm_call(
        model="gpt-4o",
        input_text=f"Solve: {task}",
        output_text="I'll break this down into steps: 1) analyze 2) execute 3) verify",
        input_tokens=random.randint(80, 200),
        output_tokens=random.randint(40, 120),
        duration_ms=duration_ms,
    )

    # ツール呼び出しをシミュレート
    t0 = time.perf_counter()
    time.sleep(0.02)
    duration_ms = (time.perf_counter() - t0) * 1000

    bb.record_tool_call(
        tool_name="web_search",
        arguments={"query": task, "max_results": 5},
        result={"results": ["result_a", "result_b", "result_c"]},
        duration_ms=duration_ms,
    )

    # 2回目の LLM 呼び出し (Claude)
    t0 = time.perf_counter()
    time.sleep(0.03)
    duration_ms = (time.perf_counter() - t0) * 1000

    bb.record_llm_call(
        model="claude-3-5-sonnet-20241022",
        input_text="Summarize search results and provide final answer.",
        output_text="Based on the research, the answer is: 42.",
        input_tokens=random.randint(150, 300),
        output_tokens=random.randint(50, 100),
        duration_ms=duration_ms,
    )

    return "Task completed."


# ============================================================
# パターン 2: コンテキストマネージャー
# ============================================================
def run_context_manager_agent(task: str) -> str:
    with BlackBox.session("context_manager_agent") as bb:
        # LLM 呼び出し
        t0 = time.perf_counter()
        time.sleep(0.04)
        bb.record_llm_call(
            model="gpt-4o-mini",
            input_text=task,
            output_text="Processing your request...",
            input_tokens=50,
            output_tokens=20,
            duration_ms=(time.perf_counter() - t0) * 1000,
        )

        # ツール呼び出し
        t0 = time.perf_counter()
        time.sleep(0.01)
        bb.record_tool_call(
            tool_name="read_file",
            arguments={"path": "/etc/hostname"},
            result={"content": "my-machine\n"},
            duration_ms=(time.perf_counter() - t0) * 1000,
        )

        return "Context manager agent done."


# ============================================================
# パターン 3: エラーが発生するケース
# ============================================================
def run_failing_agent() -> None:
    try:
        with BlackBox.session("failing_agent") as bb:
            bb.record_llm_call(
                model="gpt-4o",
                input_text="do something risky",
                output_text="attempting...",
                input_tokens=10,
                output_tokens=5,
                duration_ms=100.0,
            )
            raise ValueError("Simulated agent failure: tool returned invalid JSON")
    except ValueError:
        pass  # エラーは BlackBox が自動記録済み


# ============================================================
# main
# ============================================================
def main() -> None:
    print("\n" + "="*60)
    print("  agentblackbox — basic_usage.py")
    print("="*60)

    # --- パターン 1 ---
    print("\n[1/3] デコレータパターンで実行...")
    result1 = run_decorator_agent("What is the capital of Mars?")
    print(f"      結果: {result1}")

    # --- パターン 2 ---
    print("\n[2/3] コンテキストマネージャーパターンで実行...")
    result2 = run_context_manager_agent("Read the hostname file")
    print(f"      結果: {result2}")

    # --- パターン 3 ---
    print("\n[3/3] エラーケースのパターンで実行...")
    run_failing_agent()
    print("      (エラーが正しく記録されました)")

    # --- セッション一覧 ---
    sessions = BlackBox.list_sessions(limit=10)
    print(f"\n記録されたセッション ({len(sessions)} 件):")
    for s in sessions:
        elapsed = ""
        if s.end_time:
            elapsed = f" ({(s.end_time - s.start_time)/1e6:.0f}ms)"
        print(f"  [{s.status:8s}] {s.agent_name:<30} ${s.total_cost_usd:.6f}{elapsed}")
        print(f"             session_id: {s.session_id}")

    # --- 最新セッションを再生 ---
    if sessions:
        print("\n--- 最新セッションを replay ---")
        BlackBox.replay(sessions[0].session_id)

    # --- JSON エクスポート ---
    if sessions:
        success_sessions = [s for s in sessions if s.status == "success"]
        if success_sessions:
            exported = BlackBox.export_json(success_sessions[0].session_id)
            data = json.loads(exported)
            print(f"\nJSON エクスポートサンプル (セッション {data['session']['session_id'][:8]}...):")
            print(f"  LLM 呼び出し数: {len(data['llm_calls'])}")
            print(f"  ツール呼び出し数: {len(data['tool_calls'])}")
            print(f"  合計コスト: ${data['session']['total_cost_usd']:.6f}")

    print(f"\nデータベース: ~/.agentblackbox/recordings.db")
    print("ダッシュボードは: agentblackbox dashboard  (Phase 2)")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
