import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from google import genai

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import (
    MODEL_NAME,
    classify_intent,
    generate_grounded_reply,
    load_retriever,
    retrieve,
    trust_gate,
)

GOLD_PATH = PROJECT_ROOT / "data" / "golden_set.csv"
INDEX_DIR = PROJECT_ROOT / "data" / "retrieval_index"
RESULTS_PATH = PROJECT_ROOT / "evaluation" / "ten_case_results.jsonl"
MAX_CASES = 10
DEFAULT_SLEEP = 8.0


def load_golden():
    df = pd.read_csv(GOLD_PATH)
    text_col = "customer_text_clean"
    if text_col not in df.columns:
        raise ValueError(f"Missing {text_col}. Columns: {list(df.columns)}")
    required = [text_col, "intent_name", "should_escalate"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Golden set missing columns: {missing}")
    return df


def choose_cases(df, n):
    # Deterministic, diverse sample across intents. Prefer one example per intent,
    # then fill remaining slots by original order. This keeps the 10-case check
    # broad rather than accidentally dominated by one class.
    chosen = []
    seen = set()
    for idx, row in df.iterrows():
        intent = str(row["intent_name"])
        if intent not in seen:
            chosen.append(idx)
            seen.add(intent)
        if len(chosen) >= n:
            break
    if len(chosen) < n:
        for idx in df.index:
            if idx not in chosen:
                chosen.append(idx)
            if len(chosen) >= n:
                break
    return df.loc[chosen].copy()


def is_quota_error(exc):
    text = str(exc).lower()
    return "429" in text or "quota" in text or "too_many_requests" in text


def call_with_retry(fn, label, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            if not is_quota_error(exc) or attempt == max_retries:
                raise
            wait = 65.0
            print(f"      {label}: quota/rate-limit; waiting {wait:.0f}s (attempt {attempt}/{max_retries})")
            time.sleep(wait)


def append_result(obj):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def existing_ids():
    ids = set()
    if not RESULTS_PATH.exists():
        return ids
    for line in RESULTS_PATH.read_text(encoding="utf-8").splitlines():
        try:
            obj = json.loads(line)
            if obj.get("status") == "ok":
                ids.add(int(obj["row_index"]))
        except Exception:
            pass
    return ids


def run(limit):
    if limit < 1 or limit > MAX_CASES:
        raise ValueError(f"This final free-tier verifier intentionally supports 1-{MAX_CASES} cases only.")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured in this PowerShell session.")

    df = load_golden()
    candidates = choose_cases(df, limit)
    done = existing_ids()
    candidates = candidates[~candidates.index.isin(done)]

    print("=" * 72)
    print("10-CASE FAITHFUL PRODUCTION-AGENT VERIFICATION")
    print("=" * 72)
    print(f"Golden set available: {len(df)}")
    print(f"Cases requested: {limit}")
    print(f"Already completed: {len(done)}")
    print(f"Remaining this run: {len(candidates)}")
    print(f"Pacing: {DEFAULT_SLEEP:.1f}s between Gemini calls")
    print("The production agent is NOT modified; this script only evaluates it.")
    print()

    if not len(candidates):
        print("All selected cases are already cached.")
        return

    client = genai.Client(api_key=api_key)
    vectorizer, matrix, metadata = load_retriever(INDEX_DIR)

    for order, (row_idx, row) in enumerate(candidates.iterrows(), start=1):
        message = str(row["customer_text_clean"])
        gold_intent = str(row["intent_name"])
        gold_escalate = bool(row["should_escalate"])

        print(f"[{order}/{len(candidates)}] row={row_idx}")
        print(f"    gold intent={gold_intent} gold escalation={'ESCALATE' if gold_escalate else 'AUTO-HANDLE'}")

        try:
            intent_result = call_with_retry(
                lambda: classify_intent(client, message),
                "intent",
            )
            time.sleep(DEFAULT_SLEEP)

            cases = retrieve(message, vectorizer, matrix, metadata, top_k=5)
            generation_result = call_with_retry(
                lambda: generate_grounded_reply(client, message, intent_result, cases),
                "generation",
            )
            gate = trust_gate(intent_result, cases, generation_result)

            pred_intent = intent_result["intent"]
            pred_escalate = gate["decision"] == "ESCALATE"
            intent_correct = pred_intent == gold_intent
            escalation_correct = pred_escalate == gold_escalate

            result = {
                "status": "ok",
                "row_index": int(row_idx),
                "customer_text": message,
                "gold_intent": gold_intent,
                "predicted_intent": pred_intent,
                "intent_confidence": float(intent_result["confidence"]),
                "gold_should_escalate": gold_escalate,
                "predicted_decision": gate["decision"],
                "intent_correct": intent_correct,
                "escalation_correct": escalation_correct,
                "top_similarity": float(gate["top_similarity"]),
                "supporting_matches": int(gate["supporting_matches"]),
                "intent_reason": intent_result["reason"],
                "model_recommendation": generation_result["model_recommendation"],
                "model_reason": generation_result["model_reason"],
                "draft_reply": generation_result["draft_reply"],
                "evidence_summary": generation_result["evidence_summary"],
                "gate_reason": gate["reason"],
            }
            append_result(result)
            print(f"    pred intent={pred_intent} decision={gate['decision']} similarity={gate['top_similarity']:.3f}")
            print(f"    intent={'PASS' if intent_correct else 'FAIL'} escalation={'PASS' if escalation_correct else 'FAIL'}")

        except Exception as exc:
            print(f"    ERROR: {exc}")
            append_result({
                "status": "error",
                "row_index": int(row_idx),
                "customer_text": message,
                "error": str(exc),
            })
            if is_quota_error(exc):
                print("    Gemini quota is currently exhausted. Stop here and resume after reset.")
                break

        time.sleep(DEFAULT_SLEEP)

    print(f"\nSaved: {RESULTS_PATH}")


def metrics():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"No results found: {RESULTS_PATH}")
    rows=[]
    for line in RESULTS_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        obj=json.loads(line)
        if obj.get("status")=="ok": rows.append(obj)
    if not rows:
        print("No successful results yet.")
        return
    intent_acc=sum(r["intent_correct"] for r in rows)/len(rows)
    esc_acc=sum(r["escalation_correct"] for r in rows)/len(rows)
    auto=sum(r["predicted_decision"]=="AUTO-HANDLE" for r in rows)/len(rows)
    print("="*72)
    print("10-CASE VERIFICATION METRICS")
    print("="*72)
    print(f"Successful cases: {len(rows)}")
    print(f"Intent accuracy: {intent_acc:.4f}")
    print(f"Escalation accuracy: {esc_acc:.4f}")
    print(f"Auto-handle rate: {auto:.4f}")
    print(f"Mean intent confidence: {sum(r['intent_confidence'] for r in rows)/len(rows):.4f}")
    print(f"Mean top similarity: {sum(r['top_similarity'] for r in rows)/len(rows):.4f}")
    print("\nIMPORTANT: This is a 10-case smoke/verification result, not a claim of full 150-case performance.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--limit", type=int, default=10, choices=range(1,11), metavar="{1..10}")
    sub.add_parser("metrics")
    args = parser.parse_args()
    if args.cmd == "run": run(args.limit)
    else: metrics()
