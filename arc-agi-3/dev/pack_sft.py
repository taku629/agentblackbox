#!/usr/bin/env python3
"""Pack extracted conversations into LoRA SFT samples.

Each sample = [system] + as many preceding turns as fit (left-trimmed) +
one target assistant message at the end. One sample per assistant turn:
the model learns "given this exact game context, produce this reasoning +
action". Assistant-only label masking happens in train_lora.py.

Usage: pack_sft.py out.jsonl in1.jsonl in2.jsonl [--max-tokens 8192]
"""
import json, sys, os
from tokenizers import Tokenizer

MAX_TOKENS = 8192
TOOL_CAP = 4096  # prod tool_output cap is 4096 chars
TOK = Tokenizer.from_file(os.path.join(os.path.dirname(__file__), 'tok38/tokenizer.json'))

def norm_msg(m, src):
    m = dict(m)
    if 'reasoning' in m and 'reasoning_content' not in m:
        m['reasoning_content'] = m.pop('reasoning')
    if m.get('role') == 'tool' and len(m.get('content') or '') > TOOL_CAP:
        m['content'] = m['content'][:TOOL_CAP] + '\n...[truncated]'
    return m

def approx_text(m):
    parts = []
    if m.get('reasoning_content'):
        parts.append(m['reasoning_content'])
    parts.append(m.get('content') or '')
    for tc in m.get('tool_calls') or []:
        fn = tc.get('function') or {}
        parts.append(fn.get('name', '') + fn.get('arguments', ''))
    return '\n'.join(p for p in parts if p)

def n_tok(text):
    return len(TOK.encode(text, add_special_tokens=False).ids)

def pack_conv(conv, src, w):
    msgs = [norm_msg(m, src) for m in conv['messages']]
    if not msgs or msgs[0].get('role') != 'system':
        return 0
    sys_msg = msgs[0]
    body = msgs[1:]
    # tokenize each message once; prefix sums for O(1) window size lookups
    sys_tok = n_tok(approx_text(sys_msg)) + 8
    toks = [n_tok(approx_text(m)) + 8 for m in body]
    pref = [0]
    for t in toks:
        pref.append(pref[-1] + t)
    n = 0
    for j, m in enumerate(body):
        if m.get('role') != 'assistant':
            continue
        if not (m.get('reasoning_content') or m.get('content') or m.get('tool_calls')):
            continue
        # grow context backwards while it still fits the budget
        k = j
        while k > 0 and sys_tok + (pref[j] - pref[k-1]) + toks[j] <= MAX_TOKENS:
            k -= 1
        if sys_tok + toks[j] > MAX_TOKENS:
            continue  # even system+assistant alone won't fit
        ctx = body[k:j]
        if not any(x.get('role') == 'user' for x in ctx):
            continue  # degenerate: model always sees a user prompt in prod
        sample = [sys_msg] + ctx + [m]
        w.write(json.dumps({"game": conv['game'], "src": src,
                            "levels": conv.get('levels'),
                            "messages": sample}, ensure_ascii=False) + '\n')
        n += 1
    return n

def main():
    out = sys.argv[1]
    inputs = [a for a in sys.argv[2:] if not a.startswith('--')]
    if '--max-tokens' in sys.argv:
        global MAX_TOKENS
        i = sys.argv.index('--max-tokens')
        MAX_TOKENS = int(sys.argv[i+1])
    total = 0
    with open(out, 'w') as w:
        for f in inputs:
            src = os.path.basename(f).replace('.jsonl', '')
            for line in open(f):
                conv = json.loads(line)
                total += pack_conv(conv, src, w)
    print('samples:', total, '->', out)

if __name__ == '__main__':
    main()
