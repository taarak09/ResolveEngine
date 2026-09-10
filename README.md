#  ResolveEngine

<p align="center">
  <h3 align="center">AI-Powered Customer Support Agent</h3>
  <p align="center">
    Classify support requests • Retrieve historical evidence • Generate grounded replies • Escalate when uncertain
  </p>
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.13+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Gemini](https://img.shields.io/badge/LLM-Gemini-8E75B2?style=for-the-badge)
![scikit-learn](https://img.shields.io/badge/Retrieval-TF--IDF-F7931E?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Take--Home%20Submission-success?style=for-the-badge)

</p>

<p align="center">

**Repository:** https://github.com/taarak09/ResolveEngine

</p>

---

## 🌟 Overview

**ResolveEngine** is an AI customer-support agent developed for the **Hiver SDE Intern take-home assignment**.

The system is designed around a simple but important idea:

> **A model should not automatically answer a customer just because it is confident. It should answer when there is enough evidence to justify automation.**

For each incoming customer message, ResolveEngine:

```text
 Customer Message
        │
        ▼
 Intent Classification
        │
        ▼
 Historical Support Retrieval
        │
        ▼
 Grounded Response Generation
        │
        ▼
 Trust / Safety Gate
        │
        ├───────────────┐
        ▼               ▼
✅ AUTO-HANDLE      👤 ESCALATE

The result is a support workflow that combines LLM reasoning, historical evidence, and conservative automation.

🎯 Problem Being Solved

Given a new customer-support message, the system must answer three questions:

1. What is the customer asking about?

The message is mapped to a predefined support intent.

2. How should the customer be answered?

The system retrieves similar historical Apple Support conversations and uses them as grounding evidence for the response.

3. Should the response be sent automatically?

The system evaluates both model confidence and historical evidence before deciding between:

✅ AUTO-HANDLE

and

👤 ESCALATE


🏗️ System Architecture

                           ┌─────────────────────┐
                           │   Customer Message  │
                           └──────────┬──────────┘
                                      │
                                      ▼
                           ┌─────────────────────┐
                           │   Intent Classifier│
                           │       Gemini        │
                           └──────────┬──────────┘
                                      │
                                      ▼
                           ┌─────────────────────┐
                           │  Historical Search│
                           │       TF-IDF        │
                           └──────────┬──────────┘
                                      │
                                      ▼
                           ┌─────────────────────┐
                           │  Response Generator│
                           │       Gemini        │
                           └──────────┬──────────┘
                                      │
                                      ▼
                           ┌─────────────────────┐
                           │    Trust Gate     │
                           │ Confidence + Evidence│
                           └──────────┬──────────┘
                                      │
                           ┌──────────┴──────────┐
                           │                     │
                           ▼                     ▼
                    ┌──────────────┐      ┌──────────────┐
                    │  AUTO-HANDLE│      │  ESCALATE   │
                    └──────────────┘      └──────────────┘


🧩 Core Pipeline

ResolveEngine is implemented as four major stages.

1.  Intent Classification

Gemini classifies the incoming customer message into one of the supported intents.

The classifier returns structured information:

intent
confidence
reason

Example:

{
  "intent": "BATTERY_CHARGING",
  "confidence": 0.98,
  "reason": "The customer is reporting rapid battery drain."
}

The project uses structured outputs so that downstream code receives predictable fields rather than parsing unrestricted natural-language responses.

2.  Historical Retrieval

The response should reflect how Apple Support historically handled similar customer problems.

ResolveEngine therefore searches the historical Apple Support corpus using TF-IDF similarity.

Customer Message
       │
       ▼
TF-IDF Representation
       │
       ▼
Similarity Search
       │
       ▼
Top-K Historical Conversations

Each retrieved case contains information such as:

Customer message
Historical Apple Support response
Similarity score
Intent

These cases provide the evidence used during response generation.

3.  Grounded Response Generation

Gemini generates a draft response using:

Current Customer Message
          +
Predicted Intent
          +
Retrieved Historical Support Cases

The goal is to make the response consistent with real historical support behavior rather than allowing the model to freely invent a solution.

Conceptually:

💬 Customer
     │
     ├──────────►  Intent
     │
     └──────────►  Historical Evidence
                         │
                         ▼
                   Gemini Generator
                         │
                         ▼
                     Draft Reply
4.  Trust Gate

The final stage determines whether the response should actually be automated.

The gate evaluates:

Intent confidence
Historical similarity
Number of supporting historical examples
Generator recommendation
Ambiguity
Out-of-domain signals
High-precision product/technology signals
Normal AUTO-HANDLE policy

For normal cases, the current policy requires:

confidence >= 0.85
AND
top_similarity >= 0.45
AND
supporting_matches >= 2
AND
generator recommendation = AUTO-HANDLE

Selected high-precision product or technology cases can use a lower retrieval threshold when a strong deterministic signal is present.

ESCALATE when
 Confidence is too low
 Historical evidence is weak
 Generator recommends escalation
 Request is ambiguous
 Request is out of domain
 Intent = OTHER_OR_UNCLEAR

This gives ResolveEngine a conservative fallback instead of forcing every message through automatic handling.

🏷️ Intent Taxonomy

The current taxonomy contains 14 support categories:

ACCOUNT_AND_PAYMENT
APPS_AND_MEDIA
BATTERY_CHARGING
CALLS_MESSAGES_NOTIFICATIONS
CONNECTIVITY
DEVICE_PERFORMANCE_AND_STABILITY
HOW_TO_OR_FEATURE
IOS_UPDATE_PROBLEM
KEYBOARD_TEXT_BUG
MAC_ITUNES_DEVELOPER
ORDERS_REPAIRS_SUPPORT
OTHER_OR_UNCLEAR
SCREEN_TOUCH_DISPLAY
WATCH_AUDIO_ACCESSORIES
🧯 OTHER_OR_UNCLEAR

This is intentionally used when a message cannot be reliably assigned to a supported category or is clearly outside the target domain.

Instead of forcing a potentially incorrect intent, the system can escalate it to a human.

🍎 Dataset

The selected brand is AppleSupport.

The project uses the Customer Support on Twitter dataset and extracts historical Apple Support conversations.

Customer messages are linked to the corresponding Apple Support responses using the response relationships present in the dataset.

The project contains the following processed data:

data/
├── apple_support_pairs.csv
├── apple_support_interactions.csv
├── development_set.csv
├── golden_candidates.csv
├── golden_set.csv
└── retrieval_index/

The original twcs.csv dataset is intentionally excluded from the GitHub repository because of its large size.

📊 Evaluation Strategy

The evaluation is intentionally split into different levels rather than relying on one headline number.

                   Evaluation
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
     Trivial       Traditional    Production
     Baseline        Baseline     Verification
🥉 Majority Baseline

The majority classifier always predicts the most frequent class.

Result
Metric	Result
Accuracy	28.80%
Macro F1	3.19%

This establishes a trivial lower bound.

🥈 TF-IDF Baseline

A supervised TF-IDF classifier is trained on the development split and evaluated on a separate golden split.

Result
Metric	Result
Accuracy	41.33%
Macro F1	16.81%

This provides a simple non-LLM benchmark against which the production system can be compared.

🏆 Golden Evaluation Set

The repository contains:

data/golden_set.csv

Current size:

150 examples

There is also a separate development set so that the TF-IDF baseline does not train and evaluate on the same examples.



🧪 Production Verification

The production pipeline uses Gemini and was tested on a targeted set of 5 representative cases.

Because of the available Gemini free-tier request limit, a full 150-example live production evaluation was not performed.

Observed targeted result
Metric	Result
 Intent Accuracy	100%
 Escalation Accuracy	80%

These numbers are a targeted sanity check, not a claim of overall production accuracy.



🧾 Engineering Decision Log

The major engineering decisions were:

#	Decision
1	Selected AppleSupport as the target brand
2	Extracted customer/support pairs using response relationships
3	Created a compact support-oriented intent taxonomy
4	Added OTHER_OR_UNCLEAR as a safe fallback
5	Separated development and golden evaluation data
6	Implemented a majority-class baseline
7	Implemented a TF-IDF supervised baseline
8	Selected Gemini for production intent classification
9	Used TF-IDF for historical support retrieval
10	Used Gemini for grounded response generation
11	Added a conservative Trust Gate
12	Added high-precision product/technology signals
13	Added out-of-domain detection
14	Kept the production behavior stable after targeted verification
15	Reported limitations instead of fabricating unavailable evaluation results
🛠️ Technology Stack
Component	Technology
Language	Python 3.13+
LLM	Gemini
LLM SDK	google-genai
Structured Output	Pydantic
Retrieval	TF-IDF
ML Utilities	scikit-learn
Data Processing	Pandas / NumPy
Version Control	Git / GitHub
📁 Repository Structure
ResolveEngine/
│
├── baselines/
│   ├── majority_baseline.py
│   └── tfidf_baseline.py
│
├── configs/
│
├── data/
│   ├── apple_support_pairs.csv
│   ├── apple_support_interactions.csv
│   ├── development_set.csv
│   ├── golden_candidates.csv
│   ├── golden_set.csv
│   └── retrieval_index/
│
├── evaluation/
│   ├── verify_targeted_5.py
│   └── targeted_5_case_results.jsonl
│
├── reports/
│   └── final_report.md
│
├── src/
│   └── agent.py
│
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
📌 Important Files
src/agent.py

The main production agent.

It connects:

Gemini Classification
        +
Historical Retrieval
        +
Gemini Response Generation
        +
Trust Gate
baselines/majority_baseline.py

Implements the trivial majority-class baseline.

baselines/tfidf_baseline.py

Implements the supervised TF-IDF baseline.

evaluation/verify_targeted_5.py

Runs the real production pipeline on the targeted verification cases.

evaluation/targeted_5_case_results.jsonl

Stores the targeted production-verification results.

reports/final_report.md

Contains the detailed project report, methodology, evaluation discussion, limitations, failure modes, and future improvements.

🚀 Quick Start
1. Clone the repository
git clone https://github.com/taarak09/ResolveEngine.git
cd ResolveEngine
2. Create a virtual environment
python -m venv .venv
3. Activate the environment
.\.venv\Scripts\Activate.ps1
4. Install dependencies
pip install -r requirements.txt
5. Configure Gemini

Set the API key as an environment variable:

$env:GEMINI_API_KEY="YOUR_API_KEY"

🔐 Never commit the real API key to GitHub.

6. Run the production agent
python -m src.agent
🧪 Evaluation Commands
Production verification
python evaluation/verify_targeted_5.py
TF-IDF baseline
python baselines/tfidf_baseline.py
Majority baseline
python baselines/majority_baseline.py
🔐 Security

API credentials are not stored directly in source code.

The repository contains:

.env.example

which contains only a placeholder.

The actual Gemini API key should be supplied through an environment variable or a local .env file.

Sensitive local configuration is excluded through .gitignore.

🔬 Reproducibility

The repository contains the processed artifacts needed to reproduce the project:

requirements.txt
processed Apple Support data
retrieval index
development dataset
golden dataset
baseline scripts
evaluation scripts
production agent

A fresh environment can therefore be created without depending on the original development machine.

The original large twcs.csv dataset is intentionally not included.



🔮 Future Improvements

The next version of ResolveEngine could improve:

Current
   |
   +--> TF-IDF Retrieval
   |
   +--> Gemini Generation
   |
   +--> Trust Gate
   |
   v
Future
   |
   +--> Hybrid Semantic + Lexical Retrieval
   |
   +--> Better Evidence Ranking
   |
   +--> Human-Validated Golden Set
   |
   +--> Calibrated Confidence
   |
   +--> Validated LLM Judge
   |
   +--> Larger Production Evaluation

The long-term objective is not maximum automation.

It is:

High-confidence automation with measurable safety.

💡 Design Philosophy

ResolveEngine intentionally separates four concepts:

┌────────────────────┐
│  PREDICTION      │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│  EVIDENCE        │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│  RESPONSE        │
└─────────┬──────────┘
          ↓
┌────────────────────┐
│  TRUST DECISION  │
└─────────┬──────────┘
          ↓
     ┌────┴─────┐
     ↓          ↓
 ✅ AUTOMATE   👤 ESCALATE

The key question is not:

"Can the model answer this?"

It is:

"Do we have enough evidence to trust the answer?"

That distinction is the foundation of the system.

❤️ Final Takeaway

ResolveEngine brings together:

🧠 LLM Intent Classification

🔎 Historical Support Retrieval

✍️ Grounded Response Generation

🛡️ Conservative Trust Gating

👤 Human Escalation

📊 Explicit Evaluation

The system does not attempt to automate every customer interaction.

It automates when the intent is clear, historical evidence is strong, and the generated response passes the trust gate.

When those conditions are not satisfied, the system chooses the safer path:

👤 Escalate to a human.

Reliable automation is not about answering everything.
It is about knowing when not to answer.
