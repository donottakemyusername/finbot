from state import detect_state_events
import yfinance as yf

df = yf.download('CIEN', period='2y', interval='1d', progress=False, auto_adjust=True)
df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
df = df.dropna()

events = detect_state_events(df)
# 只看最近20个事件
for ev in events[-20:]:
    print(f"bar {ev['bar']:3d}  {df.index[ev['bar']].date()}  {ev['event']}")