"""CLI entry point: agentblackbox <command>"""
from __future__ import annotations

import sys


def main() -> None:
    args = sys.argv[1:]
    if not args:
        _print_help()
        return

    cmd = args[0]
    if cmd == "dashboard":
        _cmd_dashboard(args[1:])
    elif cmd == "sessions":
        _cmd_sessions(args[1:])
    elif cmd == "replay":
        _cmd_replay(args[1:])
    elif cmd == "export":
        _cmd_export(args[1:])
    elif cmd in ("-h", "--help", "help"):
        _print_help()
    else:
        print(f"Unknown command: {cmd}")
        _print_help()
        sys.exit(1)


def _print_help() -> None:
    print("""agentblackbox — AI agent black-box recorder

Usage:
  agentblackbox sessions            List recorded sessions
  agentblackbox replay <session_id> Replay a session in the console
  agentblackbox export <session_id> Export session as JSON
  agentblackbox dashboard           Launch the web dashboard (requires: pip install agentblackbox[dashboard])
""")


def _cmd_sessions(args: list[str]) -> None:
    from .recorder import BlackBox
    limit = 20
    sessions = BlackBox.list_sessions(limit=limit)
    if not sessions:
        print("No sessions recorded yet.")
        return
    print(f"{'Session ID':<38} {'Agent':<28} {'Status':<10} {'Cost':>10}")
    print("-" * 90)
    for s in sessions:
        print(f"{s.session_id:<38} {s.agent_name:<28} {s.status:<10} ${s.total_cost_usd:>9.6f}")


def _cmd_replay(args: list[str]) -> None:
    if not args:
        print("Usage: agentblackbox replay <session_id>")
        sys.exit(1)
    session_id = args[0]
    from .recorder import BlackBox
    bb = BlackBox.__new__(BlackBox)
    bb.session_id = session_id
    bb.agent_name = ""
    bb._db_path = None
    bb._storage = None
    bb._session = None
    bb._token = None
    bb._total_cost = 0.0
    bb.replay()


def _cmd_export(args: list[str]) -> None:
    if not args:
        print("Usage: agentblackbox export <session_id>")
        sys.exit(1)
    session_id = args[0]
    from .recorder import BlackBox
    bb = BlackBox.__new__(BlackBox)
    bb.session_id = session_id
    bb.agent_name = ""
    bb._db_path = None
    bb._storage = None
    bb._session = None
    bb._token = None
    bb._total_cost = 0.0
    print(bb.export_json())


def _cmd_dashboard(args: list[str]) -> None:
    host = "0.0.0.0"
    port = 8765
    db_path = None
    api_key = None
    i = 0
    while i < len(args):
        if args[i] == "--host" and i + 1 < len(args):
            host = args[i + 1]; i += 2
        elif args[i] == "--port" and i + 1 < len(args):
            port = int(args[i + 1]); i += 2
        elif args[i] == "--db" and i + 1 < len(args):
            db_path = args[i + 1]; i += 2
        elif args[i] == "--api-key" and i + 1 < len(args):
            api_key = args[i + 1]; i += 2
        elif args[i] == "--cloud":
            import secrets
            api_key = secrets.token_urlsafe(32)
            print(f"  Generated API key: abx_{api_key}")
            print(f"  Set AGENTBLACKBOX_API_KEY=abx_{api_key} in your agents\n")
            api_key = f"abx_{api_key}"
            i += 1
        else:
            i += 1

    try:
        from .dashboard.app import create_app
        import uvicorn
    except ImportError:
        print("Dashboard requires extra dependencies:")
        print("  pip install agentblackbox[dashboard]")
        sys.exit(1)

    app = create_app(db_path=db_path, api_key=api_key)
    mode = "cloud" if api_key else "local"
    print(f"  AgentBlackBox Dashboard [{mode}] → http://localhost:{port}")
    print(f"  Press Ctrl+C to stop\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")
