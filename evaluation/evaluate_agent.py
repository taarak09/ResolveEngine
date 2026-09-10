
from __future__ import annotations
import argparse, csv, json, os, sys, time, random
from google import genai
from pathlib import Path
from typing import Any

# Make project root importable when launched as: py evaluation\evaluate_agent.py ...
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import (
    classify_intent,
    load_retriever,
    retrieve,
    generate_grounded_reply,
    trust_gate,
)

GOLD_PATH = PROJECT_ROOT / "data" / "golden_set.csv"
RESULT_PATH = PROJECT_ROOT / "evaluation" / "agent_results.jsonl"

# Conservative pacing: agent ~= 2 Gemini calls/example.
# 7 examples/min ~= 14 requests/min, leaving margin below the user's observed 20 RPM.
SECONDS_BETWEEN_GEMINI_CALLS = 7.0
RATE_LIMIT_WAIT_MIN = 70.0
RATE_LIMIT_WAIT_MAX = 90.0
MAX_ATTEMPTS = 6

def load_gold():
    rows=[]
    with open(GOLD_PATH, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def load_cache():
    done={}
    if RESULT_PATH.exists():
        with open(RESULT_PATH, encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line: continue
                try:
                    x=json.loads(line)
                    if x.get("status")=="ok":
                        done[int(x["index"])]=x
                except Exception:
                    pass
    return done

def append_result(obj):
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False)+"\n")
        f.flush()
        os.fsync(f.fileno())

def is_rate_limit_error(e):
    s=str(e)
    return "429" in s or "quota exceeded" in s.lower() or "too_many_requests" in s.lower()

def run(limit=None, start_index=0):
    gold=load_gold()
    cache=load_cache()
    todo=[i for i in range(start_index, len(gold)) if i not in cache]
    if limit is not None:
        todo=todo[:limit]
    print("="*72)
    print("FAITHFUL PRODUCTION-AGENT EVALUATION")
    print("="*72)
    print(f"Gold examples: {len(gold)}")
    print(f"Already cached successful: {len(cache)}")
    print(f"To evaluate this run: {len(todo)}")
    print(f"Pacing: {SECONDS_BETWEEN_GEMINI_CALLS:.1f}s between Gemini calls (~{60/(SECONDS_BETWEEN_GEMINI_CALLS*2):.1f} requests/min)")
    print()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured in this PowerShell session.")
    client = genai.Client(api_key=api_key)
    retriever = load_retriever(str(PROJECT_ROOT / "data" / "retrieval_index"))
    for n, idx in enumerate(todo, start=1):
        r=gold[idx]
        text=(r.get("customer_text_clean") or "").strip()
        gold_intent=(r.get("intent_name") or "").strip()
        gold_esc=(r.get("should_escalate") or "").strip().upper()
        started=time.time()
        attempts=0
        while True:
            attempts += 1
            try:
                pred=classify_intent(client, text)
                time.sleep(SECONDS_BETWEEN_GEMINI_CALLS)
                vectorizer, matrix, metadata = retriever
                hits=retrieve(text, vectorizer, matrix, metadata, top_k=5)
                reply=generate_grounded_reply(
                    client, text, pred, hits
                )
                time.sleep(SECONDS_BETWEEN_GEMINI_CALLS)
                gate=trust_gate(pred, hits, reply)
                decision=gate["decision"]
                gate_reason=gate["reason"]
                obj={
                    "status":"ok",
                    "index":idx,
                    "customer_text":text,
                    "gold_intent":gold_intent,
                    "pred_intent":pred["intent"],
                    "intent_confidence":float(pred.get("confidence",0.0)),
                    "gold_should_escalate":gold_esc,
                    "pred_decision":decision,
                    "gate_reason":gate_reason,
                    "top_similarity":float(hits[0]["similarity"]) if hits else 0.0,
                    "retrieved_count":len(hits),
                    "draft_response":reply.get("draft_reply", ""),
                    "generator_reason":reply.get("reason",""),
                    "evidence_summary":reply.get("evidence_summary",""),
                    "attempts":attempts,
                    "elapsed_seconds":round(time.time()-started,2),
                }
                append_result(obj)
                print(f"[{idx}] OK  intent={obj['pred_intent']} gold={gold_intent} "
                      f"decision={decision} sim={obj['top_similarity']:.3f} "
                      f"({n}/{len(todo)})")
                break
            except Exception as e:
                if is_rate_limit_error(e):
                    wait=random.uniform(RATE_LIMIT_WAIT_MIN, RATE_LIMIT_WAIT_MAX)
                    print(f"[{idx}] 429/quota on attempt {attempts}; waiting {wait:.1f}s")
                    time.sleep(wait)
                    if attempts >= 6:
                        print(f"[{idx}] giving up after {attempts} attempts; will remain uncached")
                        break
                else:
                    print(f"[{idx}] ERROR: {type(e).__name__}: {e}")
                    break
        # Pace completed examples. Failed examples are not cached and can be retried by rerunning.
        time.sleep(sleep_seconds)

def f1_for_binary(tp, fp, fn):
    p=tp/(tp+fp) if tp+fp else 0.0
    r=tp/(tp+fn) if tp+fn else 0.0
    return (2*p*r/(p+r)) if p+r else 0.0, p, r

def metrics():
    cache=load_cache()
    if not cache:
        print("No successful evaluation results found.")
        return
    labels=sorted({x["gold_intent"] for x in cache.values()} | {x["pred_intent"] for x in cache.values()})
    correct=sum(x["gold_intent"]==x["pred_intent"] for x in cache.values())
    # macro F1 across labels represented in gold or predictions
    f1s=[]
    for lab in labels:
        tp=sum(x["gold_intent"]==lab and x["pred_intent"]==lab for x in cache.values())
        fp=sum(x["gold_intent"]!=lab and x["pred_intent"]==lab for x in cache.values())
        fn=sum(x["gold_intent"]==lab and x["pred_intent"]!=lab for x in cache.values())
        f1s.append(f1_for_binary(tp,fp,fn)[0])
    y=[x["gold_should_escalate"]=="YES" for x in cache.values()]
    p=[x["pred_decision"].upper()=="ESCALATE" for x in cache.values()]
    tp=sum(a and b for a,b in zip(y,p)); fp=sum((not a) and b for a,b in zip(y,p)); fn=sum(a and (not b) for a,b in zip(y,p))
    ef1,ep,er=f1_for_binary(tp,fp,fn)
    auto=sum(x["pred_decision"].upper()=="AUTO-HANDLE" for x in cache.values())/len(cache)
    c=sum(x.get("intent_confidence",0) for x in cache.values())/len(cache)
    s=sum(x.get("top_similarity",0) for x in cache.values())/len(cache)
    print("="*72)
    print("AUTOMATED AGENT EVALUATION")
    print("="*72)
    print(f"Successful examples: {len(cache)}")
    print(f"Intent accuracy: {correct/len(cache):.4f}")
    print(f"Intent macro F1: {sum(f1s)/len(f1s):.4f}")
    print(f"Escalation precision: {ep:.4f}")
    print(f"Escalation recall: {er:.4f}")
    print(f"Escalation F1: {ef1:.4f}")
    print(f"Auto-handle rate: {auto:.4f}")
    print(f"Mean intent confidence: {c:.4f}")
    print(f"Mean top similarity: {s:.4f}")

    # per-intent table
    print("\nPer-intent performance:")
    for lab in labels:
        tp=sum(x["gold_intent"]==lab and x["pred_intent"]==lab for x in cache.values())
        fp=sum(x["gold_intent"]!=lab and x["pred_intent"]==lab for x in cache.values())
        fn=sum(x["gold_intent"]==lab and x["pred_intent"]!=lab for x in cache.values())
        f1,pv,rv=f1_for_binary(tp,fp,fn)
        support=sum(x["gold_intent"]==lab for x in cache.values())
        print(f"  {lab:38s} support={support:3d} precision={pv:.3f} recall={rv:.3f} f1={f1:.3f}")

def main():
    ap=argparse.ArgumentParser()
    sub=ap.add_subparsers(dest="cmd", required=True)
    r=sub.add_parser("run")
    r.add_argument("--limit", type=int, default=None, help="number of uncached examples this run")
    r.add_argument("--start-index", type=int, default=0)
    sub.add_parser("metrics")
    args=ap.parse_args()
    if args.cmd=="run":
        run(args.limit,args.start_index)
    else:
        metrics()

if __name__=="__main__":
    main()
