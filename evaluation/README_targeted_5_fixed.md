# Fixed targeted 5-case evaluator

This evaluator fixes two issues in the earlier version:
1. Production `IntentResult` uses the field `intent`, not `intent_name`.
2. Golden `should_escalate` uses YES/NO, while the production gate uses ESCALATE/AUTO-HANDLE. The evaluator normalizes these consistently.

It calls the real production agent functions and does not modify `src/agent.py`.

Run:
```powershell
py evaluation\verify_targeted_5.py
```

It tests rows 0, 5, 6, 7 and 18 and stores results in:
`evaluation\targeted_5_case_results.jsonl`
