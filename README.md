<div align="center">

# ⚡ Resolve Engine

### Evidence-Grounded AI Customer Support Agent

**Understand → Retrieve → Generate → Verify → Escalate**

Resolve Engine is an AI-powered customer-support agent for AppleSupport that combines
LLM-based intent understanding, historical support retrieval, grounded response generation,
and evidence-aware escalation.

<br/>

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Gemini](https://img.shields.io/badge/LLM-Gemini-4285F4)](https://ai.google.dev/)
[![Retrieval](https://img.shields.io/badge/Retrieval-TF--IDF-2EA44F)](https://scikit-learn.org/)
[![Evaluation](https://img.shields.io/badge/Golden%20Set-150%20Examples-8A2BE2)](#evaluation)
[![Status](https://img.shields.io/badge/Status-Complete-2EA44F)](#)

<br/>

**Hiver SDE Intern Take-Home Project**

</div>

---

# 🌟 Project Overview

Customer-support automation is not simply a matter of generating a good-sounding reply.

A useful support agent should answer two different questions:

> **What is this customer asking about?**

and

> **Do I have enough evidence to safely answer automatically?**

**Resolve Engine** was designed around that distinction.

Built using the **Customer Support on Twitter (TWCS)** dataset, with **AppleSupport**
selected as the target brand, the system takes a new customer message and:

1. Classifies the customer's primary support intent.
2. Retrieves historically similar AppleSupport interactions.
3. Generates a response grounded in those historical cases.
4. Evaluates whether the available evidence is strong enough.
5. Decides between **AUTO-HANDLE** and **ESCALATE**.
6. Provides supporting evidence and a reason for the decision.

The core philosophy is:

> **Evidence first. Automation second.**

When the system does not have enough evidence, it is designed to escalate rather than confidently invent an answer.

---

# 🎯 Problem Statement

The project focuses on building a support agent that combines **understanding, historical precedent, response generation, and safe automation**.

A naive implementation might look like:

```text
Customer Message
       ↓
      LLM
       ↓
   Generated Reply
