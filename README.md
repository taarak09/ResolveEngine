<div align="center">

# ⚡ Resolve Engine

### Evidence-Grounded AI Customer Support Agent

**Classify → Retrieve → Generate → Verify → Escalate**

An AI support agent for AppleSupport that combines LLM reasoning with historical support evidence to draft grounded replies and avoid unsupported automation.

<br>

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![LLM](https://img.shields.io/badge/LLM-Gemini-4285F4)
![Retrieval](https://img.shields.io/badge/Retrieval-TF--IDF-2EA44F)
![Evaluation](https://img.shields.io/badge/Golden%20Set-150%20Examples-8A2BE2)
![Status](https://img.shields.io/badge/Status-Take--Home%20Complete-2EA44F)

</div>

---

# 🚀 Overview

**Resolve Engine** is an AI-powered customer-support agent built for the **AppleSupport** brand using the **Customer Support on Twitter (TWCS)** dataset.

The system is designed around a practical support workflow:

> **Understand the issue → find relevant historical resolutions → draft a grounded response → decide whether automation is trustworthy.**

Given a new customer message, Resolve Engine:

1. **Classifies** the primary support intent.
2. **Retrieves** historically similar AppleSupport interactions.
3. **Generates** a response grounded in those retrieved cases.
4. **Evaluates** the quality and strength of the available evidence.
5. **Auto-handles or escalates** the case with an explicit reason.

The main design principle is:

> **When evidence is insufficient, escalation is preferable to a confident but unsupported answer.**

This makes Resolve Engine an **evidence-aware support agent**, rather than simply a general-purpose chatbot.

---

# 🧠 System Architecture

```text
                         ┌──────────────────────┐
                         │   Customer Message   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Intent Classification│
                         │        Gemini        │
                         └──────────┬───────────┘
                                    │
                              Intent + Confidence
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Historical Retrieval │
                         │   TF-IDF + Cosine    │
                         │      Similarity      │
                         └──────────┬───────────┘
                                    │
                             Supporting Cases
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Grounded Response    │
                         │ Generation (Gemini)  │
                         └──────────┬───────────┘
                                    │
                              Draft + Recommendation
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      Trust Gate      │
                         │ Confidence + Evidence│
                         └──────────┬───────────┘
                                    │
                           ┌────────┴────────┐
                           │                 │
                           ▼                 ▼
                     AUTO-HANDLE         ESCALATE
                           │                 │
                           ▼                 ▼
                      Draft Reply       Human Review
