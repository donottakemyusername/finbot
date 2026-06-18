"""Quick batch scanner — runs trinity_analysis on a list of tickers and prints a summary table."""
import os, sys, json
from dotenv import load_dotenv

load_dotenv()

import anthropic
from tools.trinity.analysis import trinity_analysis

TICKERS = [
    "ATI","CIEN","CLS","B","INCY","ALL","COHR","WLDN","AMD","MU",
    "NBIS","PLAB","INOD","TTMI","RDW","NVDA","GLW","RMBS",
]

SIGNAL_EMOJI = {
    "strong_buy":  "[**STRONG BUY**]",
    "buy":         "[BUY]",
    "hold":        "[HOLD]",
    "sell":        "[SELL]",
    "strong_sell": "[**STRONG SELL**]",
}

def scan():
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"),
                                 base_url="https://api.anthropic.com")
    rows = []
    errors = []

    for ticker in TICKERS:
        print(f"  Analyzing {ticker}...", flush=True)
        try:
            result = trinity_analysis(ticker, client=client)
            s = result.get("summary", {})
            rows.append({
                "ticker":    ticker,
                "signal":    s.get("signal", "?"),
                "confidence":s.get("confidence","?"),
                "state":     s.get("state_label","?"),
                "entry":     s.get("entry_side","?"),
                "price":     s.get("current_price"),
                "support":   s.get("key_support"),
                "resist":    s.get("key_resistance"),
                "stop":      s.get("long_stop_loss"),
                "div":       s.get("divergence_type","none"),
                "action":    s.get("suggested_action",""),
                "risk":      s.get("key_risk",""),
                "size":      s.get("position_size",""),
            })
        except Exception as e:
            errors.append((ticker, str(e)))
            print(f"    ERROR: {e}", flush=True)

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n" + "="*90)
    print(f"{'TICKER':<8} {'SIGNAL':<22} {'CONF':<8} {'ENTRY':<8} {'PRICE':>8}  {'SUPPORT':>8}  {'RESIST':>8}  DIV")
    print("-"*90)

    # Sort: strong_buy first, then buy, then hold, then sell
    order = {"strong_buy":0,"buy":1,"hold":2,"sell":3,"strong_sell":4}
    rows.sort(key=lambda r: order.get(r["signal"],5))

    for r in rows:
        label = SIGNAL_EMOJI.get(r["signal"], r["signal"])
        price   = f"${r['price']:.2f}"   if r['price']   else "  N/A"
        support = f"${r['support']:.2f}" if r['support'] else "  N/A"
        resist  = f"${r['resist']:.2f}"  if r['resist']  else "  N/A"
        print(f"{r['ticker']:<8} {label:<22} {r['confidence']:<8} {r['entry']:<8} {price:>8}  {support:>8}  {resist:>8}  {r['div']}")

    print("="*90)

    print("\n── Actionable Details ─────────────────────────────────────────────────────────")
    for r in rows:
        if r["signal"] in ("buy","strong_buy","sell","strong_sell"):
            print(f"\n[{r['ticker']}] {SIGNAL_EMOJI.get(r['signal'])} | conf={r['confidence']} | size={r['size']}")
            print(f"  Price={r['price']:.2f}  Support={r['support']}  Resist={r['resist']}  Stop={r['stop']}")
            if r["action"]:
                print(f"  Action: {r['action']}")
            if r["risk"]:
                print(f"  Risk:   {r['risk']}")

    if errors:
        print(f"\n── Errors ({len(errors)}) ────────────────────────────────────────────")
        for t, e in errors:
            print(f"  {t}: {e}")

if __name__ == "__main__":
    scan()
