<div align="center">

# ⚡ Resolve Engine

### Evidence-Grounded AI Customer Support Agent

**Understand → Retrieve → Generate → Verify → Escalate**

An AI-powered customer-support agent for **AppleSupport** that combines LLM-based intent classification, historical support retrieval, grounded response generation, and evidence-aware escalation.

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

# 📖 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Project Goals](#-project-goals)
- [System Architecture](#-system-architecture)
- [End-to-End Workflow](#-end-to-end-workflow)
- [Dataset](#-dataset)
- [Data Preparation](#-data-preparation)
- [Intent Taxonomy](#-intent-taxonomy)
- [Historical Retrieval](#-historical-retrieval)
- [Response Generation](#-response-generation)
- [Trust & Escalation](#-trust--escalation)
- [Evaluation](#-evaluation)
- [Golden Evaluation Set](#-golden-evaluation-set)
- [Baselines](#-baselines)
- [Production Verification](#-production-verification)
- [Failure Modes](#-failure-modes)
- [What Is Misleading About the Headline Number?](#-what-is-misleading-about-the-headline-number)
- [Engineering Decisions](#-engineering-decisions)
- [Repository Structure](#-repository-structure)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Running the Agent](#-running-the-agent)
- [Evaluation & Reproducibility](#-evaluation--reproducibility)
- [Security](#-security)
- [Known Limitations](#-known-limitations)
- [What I Would Do Next](#-what-i-would-do-next)
- [Final Takeaway](#-final-takeaway)

---

# 🚀 Overview

**Resolve Engine** is an AI-powered customer-support agent built using the **Customer Support on Twitter (TWCS)** dataset, with **AppleSupport** selected as the target brand.

The system is designed to answer a practical support question:

> **Can this customer issue be safely handled automatically, and if so, what should the support agent say?**

Given a new customer message, Resolve Engine:

1. **Classifies** the customer's primary support intent.
2. **Retrieves** historically similar AppleSupport interactions.
3. **Generates** a response grounded in those historical interactions.
4. **Evaluates** whether the available evidence is strong enough for automation.
5. **Decides** between `AUTO-HANDLE` and `ESCALATE`.
6. **Provides** a reason and supporting evidence for the decision.

The core principle behind the system is:

> **Evidence first. Automation second.**

The agent is deliberately designed so that weak evidence can result in escalation instead of unsupported confidence.

---

# 🎯 Problem Statement

A naive customer-support chatbot can be implemented as:

```text
Customer Message
       ↓
      LLM
       ↓
Generated Reply
