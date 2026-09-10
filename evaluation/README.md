# 10-case faithful verification

This evaluator is intentionally capped at 10 examples for the Gemini free tier. It calls the same production functions in `src/agent.py` and does not batch, substitute a model, or alter prompts.

Run from project root:

```powershell
py evaluation\verify_10_cases.py run --limit 10
py evaluation\verify_10_cases.py metrics
```

The 10 cases are selected deterministically to maximize intent diversity. Results are checkpointed in `evaluation/ten_case_results.jsonl`.

Do not describe the resulting percentage as the performance of the full 150-example golden set. It is a targeted smoke/verification sample.
