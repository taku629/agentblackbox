"""CLI entry point: agentblackbox <command>"""
from __future__ import annotations

import sys


def main() -> None:
    args = sys.argv[1:]
    if not args:
        _print_help()
        return

    cmd = args[0]
    if cmd == "sessions":
        _cmd_sessions(args[1:])
    elif cmd == "replay":
        _cmd_replay(args[1:])
    elif cmd == "export":
        _cmd_export(args[1:])
    elif cmd == "dashboard":
        _cmd_dashboard(args[1:])
    elif cmd in ("-h", "--help", "help"):
        _print_help()
    else:
        print(f"Unknown command: {cmd}")
        _print_help()
        sys.exit(1)


def _print_help() -> None:
    print("""agentblackbox — AI agent black-box recorder

Usage:
  agentblackbox sessions [--agent NAME] [--status STATUS] [--limit N]
  agentblackbox replay <session_id>
  agentblackbox export <session_id> <output.json>
  agentblackbox dashboard [--port 8765] [--db PATH]
""")


def _cmd_sessions(args: list[str]) -> None:
    agent_name = None
    status = None
    limit = 50
    i = 0
    while i < len(args):
        if args[i] == "--agent" and i + 1 < len(args):
            agent_name = args[i + 1]; i += 2
        elif args[i] == "--status" and i + 1 < len(args):
            status = args[i + 1]; i += 2
        elif args[i] == "--limit" and i + 1 < len(args):
            limit = int(args[i + 1]); i += 2
        else:
            i += 1

    from .recorder import BlackBox
    sessions = BlackBox.list_sessions(agent_name=agent_name, status=status, limit=limit)
    if not sessions:
        print("No sessions recorded yet.")
        return
    header = f"{'ID':8s}  {'Agent':24s}  {'Status':8s}  {'Cost':>10}"
    print(header)
    print("-" * len(header))
    for s in sessions:
        print(
            f"{s.session_id[:8]}  {s.agent_name[:24]:24s}  "
            f"{s.status:8s}  ${s.total_cost_usd:>9.6f}"
        )


def _cmd_replay(args: list[str]) -> None:
    if not args:
        print("Usage: agentblackbox replay <session_id>")
        sys.exit(1)
    session_id = args[0]
    from .recorder import BlackBox, _get_storage
    session = _get_storage().get_session(session_id)
    if session is None:
        print(f"Session not found: {session_id}")
        sys.exit(1)
    bb = BlackBox.__new__(BlackBox)
    bb._session = session
    bb._token = None
    bb.agent_name = session.agent_name
    bb.replay()


def _cmd_export(args: list[str]) -> None:
    if len(args) < 2:
        print("Usage: agentblackbox export <session_id> <output.json>")
        sys.exit(1)
    session_id, output_path = args[0], args[1]
    from .recorder import BlackBox, _get_storage
    session = _get_storage().get_session(session_id)
    if session is None:
        print(f"Session not found: {session_id}")
        sys.exit(1)
    bb = BlackBox.__new__(BlackBox)
    bb._session = session
    bb._token = None
    bb.agent_name = session.agent_name
    bb.export_json(output_path)
    print(f"Exported to {output_path}")


def _cmd_dashboard(args: list[str]) -> None:
    host = "0.0.0.0"
    port = 8765
    db_path = None
    i = 0
    while i < len(args):
        if args[i] == "--host" and i + 1 < len(args):
            host = args[i + 1]; i += 2
        elif args[i] == "--port" and i + 1 < len(args):
            port = int(args[i + 1]); i += 2
        elif args[i] == "--db" and i + 1 < len(args):
            db_path = args[i + 1]; i += 2
        else:
            i += 1

    try:
        from .dashboard.app import create_app
        import uvicorn
    except ImportError:
        print("Dashboard requires extra dependencies:")
        print("  pip install agentblackbox[dashboard]")
        sys.exit(1)

    app = create_app(db_path=db_path)
    print(f"  AgentBlackBox Dashboard → http://localhost:{port}")
    print(f"  Press Ctrl+C to stop\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")
