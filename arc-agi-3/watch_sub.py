import time, sys
sys.path.insert(0,'/home/takumu/kaggle/arc-agi-3/.venv312/lib/python3.12/site-packages')
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
last = None
for _ in range(180):  # ~6h at 2min interval
    try:
        s = api.competition_submissions('arc-prize-2026-arc-agi-3')[0]
        cur = (str(s.status), getattr(s,'publicScore',None))
        if cur != last:
            print(time.strftime('%H:%M:%S'), s.ref, s.status, getattr(s,'publicScore',None), flush=True)
            last = cur
        if 'COMPLETE' in str(s.status) or 'ERROR' in str(s.status):
            break
    except Exception as e:
        print(time.strftime('%H:%M:%S'), 'ERR', str(e)[:120], flush=True)
    time.sleep(120)
