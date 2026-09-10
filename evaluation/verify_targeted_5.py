
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from google import genai
from src.agent import (
    classify_intent,
    load_retriever,
    retrieve,
    generate_grounded_reply,
    trust_gate,
)

MODEL_NAME = "gemini-3.6-flash"
GOLD_PATH = PROJECT_ROOT / "data" / "golden_set.csv"
INDEX_DIR = PROJECT_ROOT / "data" / "retrieval_index"
RESULTS_PATH = PROJECT_ROOT / "evaluation" / "targeted_5_case_results.jsonl"

# These rows target the failure modes we previously observed.
TARGET_ROWS = [0, 5, 6, 7, 18]

def load_gold():
    with GOLD_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def read_success_rows():
    done = {}
    if RESULTS_PATH.exists():
        with RESULTS_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("status") == "success":
                        done[int(r["row_index"])] = r
                except Exception:
                    pass
    return done

def save(obj):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def get_field(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)

def normalize_gold_escalation(value):
    v = str(value).strip().upper()
    if v in {"YES", "TRUE", "1", "ESCALATE"}:
        return "ESCALATE"
    if v in {"NO", "FALSE", "0", "AUTO-HANDLE", "AUTO_HANDLE"}:
        return "AUTO-HANDLE"
    raise ValueError(f"Unknown golden escalation label: {value!r}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sleep", type=float, default=8.0)
    args = p.parse_args()

    rows = load_gold()
    existing = read_success_rows()
    targets = [i for i in TARGET_ROWS if i < len(rows) and i not in existing]

    print("=" * 72)
    print("TARGETED 5-CASE FAITHFUL PRODUCTION-AGENT VERIFICATION")
    print("=" * 72)
    print(f"Golden set available: {len(rows)}")
    print(f"Target rows: {TARGET_ROWS}")
    print(f"Already completed successfully: {sorted(existing)}")
    print(f"Remaining this run: {targets}")
    print("This evaluator calls the real production agent; it does not modify src/agent.py.")
    print(f"Pacing: {args.sleep:.1f}s between Gemini calls")

    if not targets:
        print("All targeted cases already have successful results.")
        return

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY is not set in this PowerShell session.")

    client = genai.Client(api_key=api_key)
    vectorizer, matrix, metadata = load_retriever(str(INDEX_DIR))

    for n, idx in enumerate(targets, 1):
        row = rows[idx]
        msg = row.get("customer_text_clean") or row.get("customer_text") or row.get("text") or ""
        gold_intent = row.get("intent_name", "").strip()
        gold_escalation = normalize_gold_escalation(row.get("should_escalate", ""))

        print(f"\n[{n}/{len(targets)}] row={idx}")
        print(f"    gold intent={gold_intent} gold escalation={gold_escalation}")

        try:
            # Exact production sequence.
            intent_result = classify_intent(client, msg)
            pred_intent = get_field(intent_result, "intent", "")
            confidence = float(get_field(intent_result, "confidence", 0.0))
            reason = get_field(intent_result, "reason", "")

            time.sleep(args.sleep)

            cases = retrieve(msg, vectorizer, matrix, metadata, top_k=5)
            generation_result = generate_grounded_reply(
                client, msg, intent_result, cases
            )

            time.sleep(args.sleep)

            gate = trust_gate(intent_result, cases, generation_result)
            pred_decision = str(get_field(gate, "decision", "")).strip().upper()
            pred_escalation = normalize_gold_escalation(pred_decision)

            top_sim = float(cases[0]["similarity"]) if cases else 0.0
            supporting = sum(
                1 for case in cases if float(case.get("similarity", 0.0)) >= 0.35
            )

            intent_pass = pred_intent == gold_intent
            escalation_pass = pred_escalation == gold_escalation

            result = {
                "status": "success",
                "row_index": idx,
                "customer_text": msg,
                "gold_intent": gold_intent,
                "pred_intent": pred_intent,
                "intent_confidence": confidence,
                "intent_reason": reason,
                "gold_escalation": gold_escalation,
                "pred_escalation": pred_escalation,
                "top_similarity": top_sim,
                "supporting_matches": supporting,
                "generation": generation_result,
                "gate": gate,
                "intent_pass": intent_pass,
                "escalation_pass": escalation_pass,
            }
            save(result)

            print(
                f"    pred intent={pred_intent} "
                f"decision={pred_escalation} similarity={top_sim:.3f}"
            )
            print(
                f"    intent={'PASS' if intent_pass else 'FAIL'} "
                f"escalation={'PASS' if escalation_pass else 'FAIL'}"
            )

        except Exception as exc:
            text = str(exc)
            lowered = text.lower()
            if "429" in text or "quota" in lowered or "too many requests" in lowered:
                print("    GEMINI QUOTA LIMIT REACHED")
                print("    Stopping to preserve remaining quota.")
                save({
                    "status": "quota_error",
                    "row_index": idx,
                    "gold_intent": gold_intent,
                    "gold_escalation": gold_escalation,
                    "error": text,
                })
                break

            print(f"    ERROR: {text}")
            save({
                "status": "error",
                "row_index": idx,
                "gold_intent": gold_intent,
                "gold_escalation": gold_escalation,
                "error": text,
            })

    successful = read_success_rows()
    if successful:
        selected = [successful[i] for i in TARGET_ROWS if i in successful]
        if selected:
            intent_acc = sum(r["intent_pass"] for r in selected) / len(selected)
            esc_acc = sum(r["escalation_pass"] for r in selected) / len(selected)
            print("\n" + "=" * 72)
            print("TARGETED VERIFICATION SUMMARY")
            print("=" * 72)
            print(f"Successful targeted cases: {len(selected)}/{len(TARGET_ROWS)}")
            print(f"Intent accuracy: {intent_acc:.3f}")
            print(f"Escalation accuracy: {esc_acc:.3f}")
            print("These are targeted verification metrics, not a claim of 100% accuracy on all 150 cases.")

if __name__ == "__main__":
    main()
