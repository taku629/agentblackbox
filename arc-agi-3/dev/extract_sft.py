#!/usr/bin/env python3
"""Extract SFT conversations (public-dataset format) from our own run artifacts.

Input:  <artifacts_dir>/*_events.jsonl (events with `transcript` + `board` fields)
Output: sft_conversations_ours.jsonl  {game, levels, messages:[{role,content|tool_calls|tool_call_id}]}

Each analysis event's transcript is parsed into per-turn segments:
  [SYSTEM PROMPT] ... [USER PROMPT] ... [MODEL RESPONSE META] [THINKING]
  <model text> <tool_call><function=python><parameter=code>...</parameter></function></tool_call>
  [TOOL RESULT: python] ... [ANALYZER STATUS]
The conversation is stitched across events in order; the system prompt is taken
from the first event. Only games where the run reached level>1 or scored>0 are
kept by default (STaR: train on wins only). --all keeps everything.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

RE_HEADER = re.compile(r"^--- .*?---\s*$", re.M)
RE_MARK = re.compile(r"^\[(SYSTEM PROMPT|USER PROMPT|MODEL RESPONSE META|THINKING|TOOL RESULT: \w+|ANALYZER STATUS)\]\s*$", re.M)
RE_TOOLCALL = re.compile(
    r"<tool_call>\s*<function=(\w+)>\s*(.*?)\s*</function>\s*</tool_call>",
    re.S,
)
RE_PARAM = re.compile(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", re.S)

_tool_seq = 0


def parse_segments(transcript: str) -> dict[str, str]:
    """Split a transcript into labeled sections."""
    parts: dict[str, list[str]] = {}
    marks = list(RE_MARK.finditer(transcript))
    for i, m in enumerate(marks):
        name = m.group(1)
        start = m.end()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(transcript)
        parts.setdefault(name, []).append(transcript[start:end].strip())
    return parts


def parse_tool_calls(text: str) -> list[dict]:
    calls = []
    global _tool_seq
    for m in RE_TOOLCALL.finditer(text):
        fname = m.group(1)
        args = {}
        for p in RE_PARAM.finditer(m.group(2)):
            key = p.group(1).strip().lower()
            args[key] = p.group(2).strip()
        _tool_seq += 1
        calls.append(
            {
                "id": f"extracted-tool-{_tool_seq}",
                "type": "function",
                "function": {"name": fname, "arguments": json.dumps(args)},
            }
        )
    return calls


def conv_from_events(events: list[dict]) -> dict | None:
    events = sorted(events, key=lambda e: (e.get("action_num") or 0))
    system = None
    messages: list[dict] = []
    for e in events:
        tr = e.get("transcript") or ""
        if not tr:
            continue
        parts = parse_segments(tr)
        if system is None and parts.get("SYSTEM PROMPT"):
            system = parts["SYSTEM PROMPT"][0]
        for user in parts.get("USER PROMPT", []):
            if user.strip():
                messages.append({"role": "user", "content": user.strip()})
        # model response(s): each THINKING section may carry a tool_call
        thinkings = parts.get("THINKING", [])
        for think in thinkings:
            calls = parse_tool_calls(think)
            content = RE_TOOLCALL.sub("", think).strip()
            msg = {"role": "assistant", "content": content}
            if calls:
                msg["tool_calls"] = calls
            messages.append(msg)
            # tool result section (if present) pairs with the last call
            results = parts.get("TOOL RESULT: python", [])
            if calls and results:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": calls[-1]["id"],
                        "content": results[0].strip(),
                    }
                )
    if not messages:
        return None
    conv = [{"role": "system", "content": system or ""}] + messages
    return conv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("artifacts_dir")
    ap.add_argument("-o", "--out", default="sft_conversations_ours.jsonl")
    ap.add_argument("--all", action="store_true", help="keep games with score 0 too")
    args = ap.parse_args()

    out = open(args.out, "w")
    n_games = n_kept = n_msgs = 0
    for f in sorted(Path(args.artifacts_dir).glob("*_events.jsonl")):
        gid = f.name.split("-")[0]
        evs = [json.loads(l) for l in open(f)]
        analysis = [e for e in evs if e.get("type") == "analysis"]
        max_level = max((e.get("level") or 0 for e in evs), default=0)
        score = max((e.get("score") or 0 for e in evs), default=0)
        n_games += 1
        if not args.all and not (max_level > 1 or score > 0):
            continue
        conv = conv_from_events(analysis)
        if not conv:
            continue
        out.write(json.dumps({"game": gid, "levels": max_level, "messages": conv}) + "\n")
        n_kept += 1
        n_msgs += len(conv)
    out.close()
    print(f"games scanned={n_games} kept={n_kept} messages={n_msgs} -> {args.out}")


if __name__ == "__main__":
    main()
