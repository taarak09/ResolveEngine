# Targeted 5-case faithful verification

This evaluator deliberately checks five high-value cases from the existing 150-row golden set:
0, 5, 6, 7, 18.

They were chosen to cover the previously observed:
- keyboard escalation
- how-to escalation
- Mac/iTunes/developer routing
- watch/audio/accessories routing
- battery behavior

It calls the real `src.agent` functions and does not modify production code.

Run:
```powershell
py evaluation\verify_targeted_5.py
```

Results are saved to:
`evaluation\targeted_5_case_results.jsonl`
