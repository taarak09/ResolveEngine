# AppleSupport AI Support Agent

## 1. Problem Framing

### Objective

The goal of this project is to build an AI support agent for the AppleSupport brand using historical customer-support interactions from the Customer Support on Twitter (TWCS) dataset.

For each incoming customer message, the system:

1. Classifies the customer's primary support intent.
2. Retrieves historically similar AppleSupport interactions.
3. Generates a response grounded in those historical interactions.
4. Decides whether the case should be auto-handled or escalated to a human, with an explicit reason.

### Design Goal

The main design objective is **evidence-backed automation rather than maximum automation**.

A fluent language-model response is not automatically a trustworthy support response. The system therefore separates:

- understanding what the customer is asking;
- finding historical evidence for that issue;
- deciding whether that evidence is strong enough to support automatic handling.

When confidence or evidence is insufficient, the system can escalate to a human.

### AppleSupport Scope

The project focuses on AppleSupport interactions extracted from the TWCS dataset.

The current taxonomy contains 14 intents:

- ACCOUNT_AND_PAYMENT
- APPS_AND_MEDIA
- BATTERY_CHARGING
- CALLS_MESSAGES_NOTIFICATIONS
- CONNECTIVITY
- DEVICE_PERFORMANCE_AND_STABILITY
- HOW_TO_OR_FEATURE
- IOS_UPDATE_PROBLEM
- KEYBOARD_TEXT_BUG
- MAC_ITUNES_DEVELOPER
- ORDERS_REPAIRS_SUPPORT
- OTHER_OR_UNCLEAR
- SCREEN_TOUCH_DISPLAY
- WATCH_AUDIO_ACCESSORIES

### System Pipeline

```text
Customer Message
      |
      v
Intent Classification
      |
      v
Historical AppleSupport Retrieval
      |
      v
Grounded Response Generation
      |
      v
Trust / Escalation Gate
      |
      +----------------------+
      |                      |
      v                      v
 AUTO-HANDLE             ESCALATE
      |                      |
      v                      v
 Draft Reply             Human Review
```

## 2. Data and Baselines

### Historical Data

The source is the Customer Support on Twitter (TWCS) dataset. The project isolates the AppleSupport brand and constructs customer-to-support interaction pairs using the dataset's response relationships.

The resulting AppleSupport historical corpus contains approximately 106K customer-to-support pairs and forms the evidence base for retrieval.

The retrieval corpus used by the agent excludes the 150 golden evaluation examples to reduce evaluation leakage.

### Baseline 1 — Majority-Class Baseline

The first baseline predicts the most frequent intent for every example.

On the original 250-example labelled candidate set:

- Accuracy: **28.80%**
- Macro F1: **3.19%**

This provides a trivial lower baseline for the multi-class intent problem.

### Baseline 2 — TF-IDF Classifier

The second baseline uses TF-IDF features with a linear classifier.

It was trained on the 100-example development split and evaluated on the separate 150-example golden split:

- Accuracy: **41.33%**
- Macro F1: **16.81%**

This improves on the trivial baseline but remains limited by the small labelled training set, class imbalance, and semantic overlap between support categories.

### Why the Final System Uses Retrieval + LLM

The task is broader than intent classification. A useful support agent must also produce a response and determine whether automation is safe.

The final system therefore combines:

- LLM-based intent classification;
- historical retrieval;
- grounded response generation;
- evidence/confidence-based escalation.

## 3. Final System Design

### Intent Classification

Gemini is used to classify the customer's primary intent. The prompt defines the 14 allowed categories and explicit tie-breaking rules.

The system also contains high-precision routing for clearly identifiable product/platform terms such as AirPods and Mac/developer terminology. This was introduced after observing category confusion during testing.

### Historical Retrieval

For each customer message, the system retrieves the top five historical AppleSupport interactions using TF-IDF similarity.

The retrieved interactions are passed to the response generator as grounding evidence.

The retrieval layer provides both:

- the historical cases themselves;
- similarity scores that can be used by the trust gate.

### Response Generation

The response generator uses the customer's message, predicted intent, and retrieved historical cases to draft a response.

The generation rules emphasize:

- consistency with observed AppleSupport behaviour;
- requesting missing device/software details when appropriate;
- avoiding unsupported promises or policies;
- recommending escalation when evidence is insufficient.

### Trust / Escalation Gate

The trust gate combines several signals:

- intent confidence;
- historical similarity;
- supporting-case coverage;
- the generator's recommendation.

The policy is deliberately conservative:

```text
Strong intent + sufficient evidence
                |
                v
           AUTO-HANDLE

Ambiguous intent OR weak evidence
                |
                v
             ESCALATE
```

This prevents the model from treating high linguistic confidence as sufficient justification for automated customer-facing assistance.

## 4. Evaluation and Reliability

### Golden Set

The repository contains a 150-example golden evaluation set and a separate 100-example development set.

The retrieval index was constructed so that the golden examples are excluded from the retrieval corpus.

### Production Verification

Because the available Gemini free-tier project imposes a small daily request quota, a full 150-example live evaluation was not completed during development.

Instead, a targeted five-case verification was run against the **actual production agent**, without batching and without substituting another model.

The five cases were selected to exercise:

- keyboard/text bug handling;
- how-to/feature handling;
- Mac/iTunes/developer routing;
- ambiguous or unsupported requests;
- battery behaviour.

Final targeted verification:

- Intent accuracy: **5/5 (100%)**
- Escalation agreement with the current golden labels: **4/5 (80%)**

These numbers are **targeted verification results, not a claim of 100% performance on the complete golden set**.

### What the Targeted Test Demonstrated

The targeted test showed that the final agent could:

- correctly route all five selected intents;
- retrieve historical AppleSupport evidence;
- generate grounded responses;
- escalate an ambiguous/unsupported request;
- preserve conservative escalation when retrieved evidence was judged insufficient.

One battery case was classified correctly but escalated because the response generator considered the available historical evidence insufficiently specific. This illustrates a deliberate separation between intent confidence and evidence sufficiency.

### Evaluation Limitation

The headline metrics above come from a deliberately selected five-case verification set. They should not be interpreted as statistically representative of the 150-example golden set.

The repository retains the complete golden set and evaluation harness for a larger run when sufficient model/API quota is available.

### LLM-as-Judge

The intended next evaluation layer is an LLM-as-judge for response quality, using a rubric covering:

- correctness;
- groundedness;
- helpfulness;
- consistency with historical AppleSupport behaviour;
- unsupported claims;
- escalation appropriateness.

Because the free-tier quota prevented completion of the larger live evaluation, this judge was not used to produce a headline score during development. This avoids reporting an unvalidated or quota-constrained judge result as if it were a full benchmark.

## 5. Failure Modes and What They Taught Me

### Failure Mode 1 — Over-conservative similarity threshold

An earlier trust gate required a top historical similarity of 0.60 for automatic handling.

This caused legitimate cases with high intent confidence to be escalated unnecessarily. An Auto-Brightness example had a similarity around 0.559 while still having relevant supporting cases.

Lesson: a single lexical similarity threshold is too crude. Retrieval evidence should be combined with intent confidence, supporting-case coverage, and the generator's recommendation.

### Failure Mode 2 — Mac/developer intent confusion

An earlier evaluation mapped an iMac/Photoshop problem to DEVICE_PERFORMANCE_AND_STABILITY instead of MAC_ITUNES_DEVELOPER.

The overlap between general performance language and platform/tool-specific language caused the error.

Lesson: high-precision platform/tool signals can reduce repeatable category confusion.

### Failure Mode 3 — Accessory intent confusion

An earlier evaluation mapped an AirPods-related request to ORDERS_REPAIRS_SUPPORT.

The system was over-weighting generic problem language such as “connection” and “issue”.

Lesson: strong product identifiers should receive sufficient weight during routing.

A subsequent production test correctly classified:

```text
My AirPods are not connecting to my iPhone
```

as WATCH_AUDIO_ACCESSORIES with high confidence.

### Failure Mode 4 — Correct intent does not guarantee safe automation

The battery-after-update example was correctly classified as BATTERY_CHARGING, but the generator recommended escalation because the retrieved evidence was considered insufficiently specific.

Lesson:

```text
Correct intent
    !=
Sufficient evidence for automatic response
```

This distinction is an intentional part of the final design.

### Failure Mode 5 — Ambiguous or unsupported requests

The message:

```text
Hello I founded an opening in your website how I can report it?
```

was treated as OTHER_OR_UNCLEAR and escalated.

Lesson: the system should not force every message into a specific product-support category. Unsupported or ambiguous requests should remain reviewable by a human.

## 6. What Is Misleading About My Headline Number?

The most tempting headline number is:

> **100% intent accuracy on the five-case targeted verification.**

That number is useful as a sanity check, but it is not evidence that the entire agent is 100% accurate.

The five cases were intentionally selected to cover known failure modes rather than sampled uniformly from the 150-example golden set.

Therefore:

```text
100% on 5 targeted cases
            !=
100% on all 150 cases
            !=
100% on future customer messages
```

The result does demonstrate that the final production pipeline handled the selected categories correctly and that the routing changes fixed the observed intent-confusion cases.

The escalation result was 80% agreement with the current golden labels. The remaining disagreement was a conservative escalation on a battery case where the generator judged the retrieved evidence insufficiently specific.

I would therefore avoid presenting a single “overall accuracy” number for the complete support agent. Intent classification, response quality, retrieval quality, and safe automation are distinct dimensions.

## 7. What I Would Do Next in One Week

### Day 1 — Human-verify the benchmark

Independently review the 150 golden examples using explicit intent and escalation guidelines.

Resolve ambiguous labels before using the set as the final benchmark.

### Day 2 — Improve intent classification

Use the expanded confusion matrix to identify the most frequently confused intent pairs.

Add or revise routing rules only when a repeatable pattern is supported by the data.

### Day 3 — Improve retrieval

Compare TF-IDF against a semantic or hybrid retrieval approach.

Measure whether better retrieval increases evidence quality on difficult cases.

### Day 4 — Add response claim verification

Add a response verifier that checks whether important factual/policy claims in the generated draft are supported by retrieved historical evidence.

Unsupported claims should trigger escalation.

### Day 5 — Calibrate the trust gate

Use the development set to tune the trade-off between automation coverage and incorrect/unsupported automation.

Do not tune thresholds on the golden test set.

### Day 6 — Full evaluation

Run the complete golden benchmark with sufficient model quota and measure:

- intent accuracy;
- macro F1;
- escalation precision/recall/F1;
- auto-handle coverage;
- retrieval quality;
- response groundedness;
- response helpfulness.

Validate the LLM-as-judge against human ratings before using it for headline reporting.

### Day 7 — Production readiness

Finalize:

- reproducible setup;
- secret/environment-variable handling;
- structured logs;
- evaluation artifacts;
- failure analysis;
- a lightweight demo interface;
- documentation for running the system on a clean machine.

## 8. Reproducibility

A clean environment can install the project's dependencies with:

```bash
pip install -r requirements.txt
```

Set the API key using an environment variable:

```text
GEMINI_API_KEY=YOUR_API_KEY
```

Then run:

```bash
py src/agent.py --message "My iPhone battery is draining very quickly after the latest update"
```

The repository does not require machine-specific `C:\Users\...` paths.

The production agent also handles Gemini quota/rate-limit failures with a user-facing message instead of exposing a Python traceback.

## 9. Conclusion

This project demonstrates an evidence-backed AI support workflow rather than a generic chatbot.

The central design principle is:

> **Understand the customer confidently, but automate conservatively.**

The system combines LLM intent classification, historical AppleSupport retrieval, grounded response generation, and an explicit trust/escalation gate.

The strongest next improvement is response-level evidence verification, followed by a larger human-validated evaluation.
