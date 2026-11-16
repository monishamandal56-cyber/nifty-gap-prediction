"""
daily_predict.py
----------------
Runs the Nifty 50 gap prediction and saves the output to:
    logs/predictions/YYYY-MM-DD.json    (structured, for apps)
    logs/predictions/YYYY-MM-DD.txt     (human-readable, for logs)
    logs/predictions/YYYY-MM-DD.md      (markdown, for GitHub viewing)
    logs/predictions/latest.json        (always = latest)
    logs/predictions/latest.md          (always = latest, for phone)

All text files use UTF-8 encoding (required for emoji in Markdown).

Optional: sends email via Gmail (if credentials configured via env vars).
"""

import os
import sys
import json
import traceback
import datetime as dt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.predictor import predict_tomorrow


# ============================================================
# OPTIONAL: Email config
# ============================================================

EMAIL_ENABLED    = False
SENDER_EMAIL     = os.environ.get("SENDER_EMAIL",    "your_email@gmail.com")
APP_PASSWORD     = os.environ.get("APP_PASSWORD",    "abcd efgh ijkl mnop")
RECIPIENT_EMAIL  = os.environ.get("RECIPIENT_EMAIL", "your_email@gmail.com")

if os.environ.get("SENDER_EMAIL") and os.environ.get("APP_PASSWORD"):
    EMAIL_ENABLED = True


LOG_DIR = os.path.join(PROJECT_ROOT, "logs", "predictions")
os.makedirs(LOG_DIR, exist_ok=True)


# ============================================================
# FORMATTERS
# ============================================================

def fmt_text(res):
    lines = [
        "=" * 55,
        "NIFTY 50 GAP PREDICTION",
        f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 55,
        f"Last close    : {res['last_close']:.2f}  ({res['date_of_last_close']})",
        f"Predicted gap : {res['gap_pct']:+.2f}%  ({res['gap_points']:+.1f} pts)",
        f"Predicted open: {res['predicted_open']:.2f}",
        f"80% band      : {res['expected_low']:.2f}  to  {res['expected_high']:.2f}",
        f"Direction     : {res['direction']}",
        f"Probabilities : Up {res['prob_up']}%  |  Flat {res['prob_flat']}%  |  Down {res['prob_down']}%",
        "",
        "STRANGLE SETUP (if inside band at 9:20 AM):",
        f"  Sell Put  at {res['expected_low'] - 50:.0f}",
        f"  Sell Call at {res['expected_high'] + 80:.0f}",
        "",
        "Top drivers:",
    ]
    for d in res.get("drivers_formatted", [])[:5]:
        sign = "+" if d["impact_pct"] >= 0 else "-"
        lines.append(f"  {d['name']:<32s} {sign}{abs(d['impact_pct']):.3f}%")

    if res.get("validation_metrics"):
        m = res["validation_metrics"]
        lines += [
            "",
            "Historical validation:",
            f"  OOF MAE      : {m.get('mae_pct', 'N/A')}%",
            f"  OOF DirAcc   : {m.get('directional_accuracy_pct', 'N/A')}%",
            f"  CI80 coverage: {m.get('ci80_empirical_coverage_pct', 'N/A')}%",
        ]

    lines.append("=" * 55)
    return "\n".join(lines)


def fmt_markdown(res):
    direction = res["direction"]
    if "UP" in direction:
        emoji = "🟢"
    elif "DOWN" in direction:
        emoji = "🔴"
    else:
        emoji = "⚪"

    lines = [
        f"# {emoji} Nifty Gap Prediction",
        "",
        f"**Generated:** {dt.datetime.now().strftime('%Y-%m-%d %H:%M IST')}",
        "",
        "---",
        "",
        f"## {res['gap_pct']:+.2f}% ({res['gap_points']:+.1f} pts)",
        f"### {direction}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| **Predicted open** | {res['predicted_open']:.2f} |",
        f"| **80% band** | {res['expected_low']:.0f} - {res['expected_high']:.0f} |",
        f"| **Last close** | {res['last_close']:.2f} ({res['date_of_last_close']}) |",
        "",
        "### Probabilities",
        "",
        "| Up | Flat | Down |",
        "|---|---|---|",
        f"| {res['prob_up']}% | {res['prob_flat']}% | {res['prob_down']}% |",
        "",
        "### Strangle Setup",
        "",
        "```",
        f"Sell Put  at {res['expected_low'] - 50:.0f}",
        f"Sell Call at {res['expected_high'] + 80:.0f}",
        "```",
        "",
        "### Top Drivers",
        "",
    ]

    for d in res.get("drivers_formatted", [])[:5]:
        sign = "🟢 +" if d["impact_pct"] >= 0 else "🔴 -"
        lines.append(f"- **{d['name']}**: {sign}{abs(d['impact_pct']):.3f}%")

    if res.get("validation_metrics"):
        m = res["validation_metrics"]
        lines += [
            "",
            "### Historical Validation",
            "",
            f"- OOF MAE: **{m.get('mae_pct', 'N/A')}%**",
            f"- OOF DirAcc: **{m.get('directional_accuracy_pct', 'N/A')}%**",
            f"- CI80 coverage: {m.get('ci80_empirical_coverage_pct', 'N/A')}%",
        ]

    lines += [
        "",
        "---",
        "",
        "*Check at 8:30 AM IST before 9:15 market open.*",
        "*Not financial advice. Trade at your own risk.*",
    ]
    return "\n".join(lines)


# ============================================================
# EMAIL (optional)
# ============================================================

def send_email(subject, body):
    import smtplib
    from email.message import EmailMessage

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"]    = SENDER_EMAIL
    msg["To"]      = RECIPIENT_EMAIL
    msg.set_content(body)

    with smtplib.SMTP("smtp.gmail.com", 587) as s:
        s.ehlo()
        s.starttls()
        s.ehlo()
        s.login(SENDER_EMAIL, APP_PASSWORD)
        s.send_message(msg)


# ============================================================
# MAIN
# ============================================================

def main():
    today     = dt.date.today().isoformat()
    json_path = os.path.join(LOG_DIR, f"{today}.json")
    text_path = os.path.join(LOG_DIR, f"{today}.txt")
    md_path   = os.path.join(LOG_DIR, f"{today}.md")
    latest_j  = os.path.join(LOG_DIR, "latest.json")
    latest_m  = os.path.join(LOG_DIR, "latest.md")

    print(f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] Starting daily prediction ...")

    # ---- Run prediction ----
    try:
        res = predict_tomorrow()
    except Exception:
        error_text = (
            f"PREDICTION FAILED at {dt.datetime.now()}\n\n"
            f"{traceback.format_exc()}"
        )
        err_path = os.path.join(LOG_DIR, f"{today}_error.txt")
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(error_text)
        print(error_text)

        if EMAIL_ENABLED:
            try:
                send_email(f"[ERROR] Nifty Bot {today}", error_text)
            except Exception as e2:
                print(f"Email also failed: {e2}")
        sys.exit(1)

    # ---- Save JSON ----
    serializable = {k: v for k, v in res.items() if k != "live_snapshot"}
    serializable["generated_at"] = dt.datetime.now().isoformat()

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, default=str, ensure_ascii=False)
    with open(latest_j, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, default=str, ensure_ascii=False)

    # ---- Save text ----
    text_body = fmt_text(res)
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text_body)

    # ---- Save markdown ----
    md_body = fmt_markdown(res)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_body)
    with open(latest_m, "w", encoding="utf-8") as f:
        f.write(md_body)

    print(text_body)
    print(f"\nSaved: {json_path}")
    print(f"Saved: {text_path}")
    print(f"Saved: {md_path}")
    print(f"Saved: {latest_m}")

    # ---- Optional email ----
    if EMAIL_ENABLED:
        subject = f"Nifty Gap {today}: {res['gap_pct']:+.2f}% ({res['direction']})"
        try:
            send_email(subject, text_body)
            print("Email sent OK")
        except Exception as e:
            print(f"Email failed (non-fatal): {e}")


if __name__ == "__main__":
    main()