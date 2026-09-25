#!/usr/bin/env python3
"""Extract STaR SFT conversations from our winner-run events.jsonl transcripts.

Output: one JSONL line per game:
  {game, levels, messages:[{role, system|user|assistant|tool, content,
                            reasoning_content?, tool_calls?, tool_call_id?}]}

Format matches the public justforgags arc3-sft-trajectories dataset plus a
reasoning_content field so Qwen3.8's chat template renders <think> blocks.

Transcript structure per analysis event:
  [SYSTEM PROMPT] once, then per model cycle:
    [MODEL RESPONSE META]  (has raw_tool_calls JSON)
    [THINKING]             (reasoning text)
    [ASSISTANT]            (optional world-model note text)
    [TOOL CALL: python]    (display markup; skipped, meta has structured calls)
    [TOOL RESULT: python]  (tool output -> role=tool)
    [ANALYZER STATUS]      (skipped)
"""
import json, re, sys, glob, os

RE_MARK = re.compile(r'^\[([A-Z][^\]\n]*)\]\s*$', re.M)
SECTION_SKIP = {"TOOL CALL: PYTHON", "ANALYZER STATUS", "MODEL RESPONSE META"}

def parse_meta_tool_calls(meta_text):
    m = re.search(r'raw_tool_calls:\s*(\[.*?\n\s*\])', meta_text, re.S)
    if not m:
        return []
    try:
        calls = json.loads(m.group(1))
    except Exception:
        return []
    out = []
    for i, c in enumerate(calls):
        fn = (c.get("function") or {})
        out.append({
            "id": c.get("id") or f"extracted-tool-{i}",
            "type": "function",
            "function": {"name": fn.get("name", "python"),
                          "arguments": fn.get("arguments", "{}")},
        })
    return out

def sections(tr):
    marks = list(RE_MARK.finditer(tr))
    for i, m in enumerate(marks):
        end = marks[i+1].start() if i+1 < len(marks) else len(tr)
        yield m.group(1).strip().upper(), tr[m.end():end]

def extract_game(path):
    evs = [json.loads(l) for l in open(path)]
    game = os.path.basename(path).split('-')[0]
    max_lv = 0
    msgs = []
    sys_done = False
    pending = None  # accumulating assistant msg for current cycle
    tc = 0
    for e in evs:
        max_lv = max(max_lv, e.get('level') or 0)
        tr = e.get('transcript') or ''
        if not tr:
            continue
        for name, body in sections(tr):
            if name == 'SYSTEM PROMPT':
                if not sys_done:
                    msgs.append({"role": "system", "content": body.strip()})
                    sys_done = True
            elif name == 'USER PROMPT':
                if pending is not None:
                    msgs.append(pending)
                    pending = None
                msgs.append({"role": "user", "content": body.strip()})
            elif name == 'MODEL RESPONSE META':
                calls = parse_meta_tool_calls(body)
                pending = {"role": "assistant", "reasoning_content": "",
                           "content": "", "tool_calls": calls}
                tc = 0
            elif name == 'THINKING':
                if pending is None:
                    pending = {"role": "assistant", "reasoning_content": "",
                               "content": "", "tool_calls": []}
                pending["reasoning_content"] = body.strip()
            elif name == 'ASSISTANT':
                if pending is None:
                    pending = {"role": "assistant", "reasoning_content": "",
                               "content": "", "tool_calls": []}
                pending["content"] = body.strip()
            elif name.startswith('TOOL RESULT'):
                if pending is not None:
                    msgs.append(pending)
                    pending = None
                call_id = None
                if msgs and msgs[-1].get('role') == 'assistant' and msgs[-1].get('tool_calls'):
                    if tc < len(msgs[-1]['tool_calls']):
                        call_id = msgs[-1]['tool_calls'][tc]['id']
                        tc += 1
                tm = {"role": "tool", "content": body.strip()}
                if call_id:
                    tm["tool_call_id"] = call_id
                msgs.append(tm)
    if pending is not None:
        msgs.append(pending)
    # drop empties
    for m in msgs:
        if m.get('role') == 'assistant':
            if not m['reasoning_content']:
                del m['reasoning_content']
            if not m['tool_calls']:
                del m['tool_calls']
    return {"game": game, "levels": max_lv, "messages": msgs}

def main():
    artifacts = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else 'sft_ours.jsonl'
    keep_all = '--all' in sys.argv
    n_scan = n_keep = n_msg = 0
    with open(out, 'w') as w:
        for f in sorted(glob.glob(os.path.join(artifacts, '*_events.jsonl'))):
            conv = extract_game(f)
            n_scan += 1
            if not keep_all and conv['levels'] <= 1:
                continue
            n_keep += 1
            n_msg += len(conv['messages'])
            w.write(json.dumps(conv, ensure_ascii=False) + '\n')
    print(f'games scanned={n_scan} kept={n_keep} messages={n_msg} -> {out}')

if __name__ == '__main__':
    main()
