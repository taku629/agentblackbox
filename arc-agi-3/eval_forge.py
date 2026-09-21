"""Evaluate Forge (frame-only mode) across all local games, mirroring hidden-eval conditions."""
import sys, importlib.util, logging, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent/"ARC-AGI-3-Agents"))
import arc_agi
from arc_agi import OperationMode

MAX_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 500
GAMES = sys.argv[2].split(",") if len(sys.argv) > 2 else None

logging.basicConfig(level=logging.ERROR)
spec = importlib.util.spec_from_file_location("forge", str(Path(__file__).resolve().parent/"forge_agent_evalmode.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
Cls = mod.MyAgent
Cls.MAX_ACTIONS = MAX_STEPS

arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
envs = arc.get_environments()
ids = [e.game_id.split("-")[0] for e in envs]
if GAMES:
    ids = [g for g in ids if g in GAMES]

results = []
for gid in ids:
    t0 = time.time()
    try:
        env = arc.make(gid)
        agent = Cls(card_id="eval", game_id=gid, agent_name=f"forge.{gid}",
                    ROOT_URL="http://localhost", record=False, arc_env=env, tags=[])
        agent.main()
        f = agent.frames[-1]
        results.append((gid, f.levels_completed, agent.action_counter, str(f.state), round(time.time()-t0)))
        print(f"{gid}: levels={f.levels_completed} actions={agent.action_counter} state={f.state} {time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        results.append((gid, 0, 0, f"ERR:{e}", 0))
        print(f"{gid}: ERROR {e}", flush=True)
        traceback.print_exc()

n = sum(1 for r in results if r[1] > 0)
tot = sum(r[1] for r in results)
print(f"\nFORGE-EVAL: {n}/{len(results)} games nonzero, {tot} levels total")
