
import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from typing import Literal
from scipy.sparse import load_npz
from google import genai


MODEL_NAME = "gemini-3.6-flash"

INTENTS = [
    "ACCOUNT_AND_PAYMENT",
    "APPS_AND_MEDIA",
    "BATTERY_CHARGING",
    "CALLS_MESSAGES_NOTIFICATIONS",
    "CONNECTIVITY",
    "DEVICE_PERFORMANCE_AND_STABILITY",
    "HOW_TO_OR_FEATURE",
    "IOS_UPDATE_PROBLEM",
    "KEYBOARD_TEXT_BUG",
    "MAC_ITUNES_DEVELOPER",
    "ORDERS_REPAIRS_SUPPORT",
    "OTHER_OR_UNCLEAR",
    "SCREEN_TOUCH_DISPLAY",
    "WATCH_AUDIO_ACCESSORIES",
]

INTENT_DEFINITIONS = {
    "ACCOUNT_AND_PAYMENT":
        "Apple ID, password, Apple Pay, billing, subscriptions, or payment problems.",
    "APPS_AND_MEDIA":
        "Apps, app installation, Apple Music, media playback, or app-specific problems.",
    "BATTERY_CHARGING":
        "Battery drain, battery health, charging, charging speed, or charging hardware.",
    "CALLS_MESSAGES_NOTIFICATIONS":
        "Phone calls, voicemail, iMessage/SMS, or notifications.",
    "CONNECTIVITY":
        "Wi-Fi, Bluetooth, cellular, VPN, or other network/connectivity problems.",
    "DEVICE_PERFORMANCE_AND_STABILITY":
        "General device slowness, freezing, crashing, lag, overheating, or instability when no clearer specialized intent applies.",
    "HOW_TO_OR_FEATURE":
        "A general how-to, settings, feature-availability, or feature-explanation question.",
    "IOS_UPDATE_PROBLEM":
        "Problems installing/updating iOS, update availability, or questions specifically about performing an update.",
    "KEYBOARD_TEXT_BUG":
        "Keyboard, typing, autocorrect, text-entry, character rendering, or question-mark/box text bugs.",
    "MAC_ITUNES_DEVELOPER":
        "macOS/Mac, iTunes, Xcode, developer tools, or related Mac/developer issues.",
    "ORDERS_REPAIRS_SUPPORT":
        "Orders, reservations, repairs, service appointments, or an existing support case.",
    "OTHER_OR_UNCLEAR":
        "Insufficient information, irrelevant/noisy follow-up, or no reasonable matching intent.",
    "SCREEN_TOUCH_DISPLAY":
        "Screen, display, touch, black-screen, or physical screen issues.",
    "WATCH_AUDIO_ACCESSORIES":
        "Apple Watch, AirPods, headphones, or other accessories.",
}


class IntentResult(BaseModel):
    intent: Literal[
        "ACCOUNT_AND_PAYMENT",
        "APPS_AND_MEDIA",
        "BATTERY_CHARGING",
        "CALLS_MESSAGES_NOTIFICATIONS",
        "CONNECTIVITY",
        "DEVICE_PERFORMANCE_AND_STABILITY",
        "HOW_TO_OR_FEATURE",
        "IOS_UPDATE_PROBLEM",
        "KEYBOARD_TEXT_BUG",
        "MAC_ITUNES_DEVELOPER",
        "ORDERS_REPAIRS_SUPPORT",
        "OTHER_OR_UNCLEAR",
        "SCREEN_TOUCH_DISPLAY",
        "WATCH_AUDIO_ACCESSORIES",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class GenerationResult(BaseModel):
    draft_reply: str
    evidence_summary: str
    model_recommendation: Literal["AUTO-HANDLE", "ESCALATE"]
    model_reason: str


class GeminiQuotaError(RuntimeError):
    """Raised when the Gemini API rate/daily quota blocks a request."""


# High-precision deterministic anchors for unambiguous AppleSupport intents.
# These are applied before Gemini so clear product/tool mentions do not get
# confused with generic performance or order/support intents.
HIGH_PRECISION_ANCHORS = {
    "MAC_ITUNES_DEVELOPER": [
        "xcode", "swift", "swiftui", "developer", "developer tools",
        "macos", "macbook", "imac", "mac mini", "mac studio", "itunes",
    ],
    "WATCH_AUDIO_ACCESSORIES": [
        "airpods", "airpod", "apple watch", "applewatch", "beats",
        "earpods", "headphones", "earbuds",
    ],
}

# Generic patterns for messages that are not well represented by the
# support taxonomy. These are intentionally conservative and route to
# OTHER_OR_UNCLEAR instead of inventing a more specific Apple intent.
def low_information_or_out_of_domain_intent(message):
    text = message.lower()

    # Website/security/reporting requests do not map cleanly to the 14
    # historical support intents. Keep them human-reviewed.
    if "website" in text and any(
        token in text for token in ["report", "issue", "problem", "opening", "vulnerability", "bug"]
    ):
        return {
            "intent": "OTHER_OR_UNCLEAR",
            "confidence": 0.97,
            "reason": "The message is about reporting an issue on a website rather than a supported Apple product intent.",
        }

    # Clearly non-Apple device requests should not be forced into a nearby intent.
    if any(device in text for device in ["samsung", "galaxy", "android", "pixel"]) and not any(
        apple in text for apple in ["iphone", "ipad", "mac", "apple watch", "airpods"]
    ):
        return {
            "intent": "OTHER_OR_UNCLEAR",
            "confidence": 0.97,
            "reason": "The request appears to target a non-Apple device and does not match the AppleSupport taxonomy.",
        }

    return None


def high_precision_intent(message):
    text = message.lower()
    hits = []
    for intent, anchors in HIGH_PRECISION_ANCHORS.items():
        for anchor in anchors:
            if anchor in text:
                hits.append((intent, anchor))
    if not hits:
        return None

    # Prefer the most specific anchor match. If multiple product families
    # appear, defer to Gemini rather than forcing a potentially wrong label.
    intents = {intent for intent, _ in hits}
    if len(intents) != 1:
        return None

    intent, anchor = hits[0]
    return {
        "intent": intent,
        "confidence": 0.99,
        "reason": f"High-precision product/tool anchor detected: '{anchor}'.",
    }


def call_structured(client, prompt, schema):
    try:
        interaction = client.interactions.create(
            model=MODEL_NAME,
            input=prompt,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": schema.model_json_schema(),
            },
        )
    except Exception as exc:
        msg = str(exc)
        lowered = msg.lower()
        if ("429" in lowered or "too many requests" in lowered or
                "quota exceeded" in lowered or "rate limit" in lowered):
            raise GeminiQuotaError(
                "Gemini API limit reached. Please try again later. "
                "Your saved project/data are unchanged."
            ) from exc
        raise

    return schema.model_validate_json(interaction.output_text)


def classify_intent(client, message):
    # Handle clearly out-of-taxonomy requests before asking Gemini.
    out_of_domain = low_information_or_out_of_domain_intent(message)
    if out_of_domain is not None:
        return out_of_domain

    intents_text = "\n".join(
        f"- {name}: {INTENT_DEFINITIONS[name]}" for name in INTENTS
    )

    prompt = f"""
Classify this AppleSupport customer-support message.

CUSTOMER MESSAGE:
{message}

ALLOWED INTENTS:
{intents_text}

Rules:
1. Choose exactly one intent.
2. Choose the customer's primary problem, not merely a keyword.
3. If an iOS update explicitly caused a specialized problem such as battery drain,
   freezing, overheating, calls, or Wi-Fi, prefer that specialized problem intent
   when clearly primary.
4. Use IOS_UPDATE_PROBLEM when the update/install itself is the problem or the user
   is asking about performing an update.
5. Use OTHER_OR_UNCLEAR when the message is vague, only a follow-up, is about an
   unsupported/non-Apple request, or lacks enough information to map reliably.
6. Confidence is your calibrated confidence in the chosen label from 0 to 1.
7. Return only the requested structured JSON.
"""

    anchored = high_precision_intent(message)
    if anchored is not None:
        return anchored

    result = call_structured(client, prompt, IntentResult)

    return {
        "intent": result.intent,
        "confidence": float(result.confidence),
        "reason": result.reason.strip(),
    }


def load_retriever(index_dir):
    index_dir = Path(index_dir)
    vectorizer = joblib.load(index_dir / "tfidf_vectorizer.joblib")
    matrix = load_npz(index_dir / "tfidf_matrix.npz")
    metadata = pd.read_csv(index_dir / "retrieval_metadata.csv")
    return vectorizer, matrix, metadata


def retrieve(message, vectorizer, matrix, metadata, top_k):
    query_vector = vectorizer.transform([message])
    scores = np.asarray((matrix @ query_vector.T).toarray()).ravel()

    indices = np.argsort(scores)[::-1][:min(top_k, len(metadata))]

    cases = []
    for idx in indices:
        row = metadata.iloc[idx]
        cases.append({
            "similarity": float(scores[idx]),
            "customer": str(row["customer_text_clean"]),
            "support": str(row["support_text_clean"]),
        })
    return cases


def generate_grounded_reply(client, message, intent_result, cases):
    evidence = "\n\n".join(
        f"""HISTORICAL CASE {i}
Similarity: {case['similarity']:.4f}
Customer:
{case['customer']}
AppleSupport:
{case['support']}"""
        for i, case in enumerate(cases, 1)
    )

    prompt = f"""
Draft an AppleSupport reply for the current customer.

CURRENT CUSTOMER:
{message}

PREDICTED INTENT:
{intent_result["intent"]}

CLASSIFIER CONFIDENCE:
{intent_result["confidence"]:.4f}

HISTORICAL EVIDENCE:
{evidence}

Rules:
1. Base the reply primarily on the historical AppleSupport evidence.
2. Do not invent policies, prices, timelines, warranties, refunds, account actions,
   or troubleshooting steps not supported by the evidence.
3. Do not claim an action was taken unless supported.
4. If evidence is weak, contradictory, or insufficient, recommend escalation.
5. Be concise, professional, and customer-facing.
6. Do not mention that you are an AI.
7. Return only the requested structured JSON.
"""

    result = call_structured(client, prompt, GenerationResult)

    return {
        "draft_reply": result.draft_reply.strip(),
        "evidence_summary": result.evidence_summary.strip(),
        "model_recommendation": result.model_recommendation,
        "model_reason": result.model_reason.strip(),
    }


def trust_gate(intent_result, cases, generation_result):
    top_similarity = cases[0]["similarity"] if cases else 0.0
    supporting_matches = sum(
        1 for case in cases if case["similarity"] >= 0.35
    )

    intent = intent_result["intent"]
    confidence = float(intent_result["confidence"])
    generator_recommends_auto = generation_result["model_recommendation"] == "AUTO-HANDLE"
    is_high_precision = (
        confidence >= 0.99
        and (
            "High-precision product/tool anchor detected:" in
            str(intent_result.get("reason", ""))
        )
    )

    reasons = []

    # Explicitly ambiguous/out-of-taxonomy requests should always be reviewed.
    if intent == "OTHER_OR_UNCLEAR":
        reasons.append("intent is unclear or outside the supported taxonomy")

    if confidence < 0.85:
        reasons.append(
            f"intent confidence {confidence:.2f} is below 0.85"
        )

    if not cases:
        reasons.append("no historical evidence was retrieved")

    # A generator escalation recommendation is a hard safety stop.
    if not generator_recommends_auto:
        reasons.append("response generator recommended escalation")

    # Balanced evidence policy:
    # - Normal intents need reasonably strong retrieval plus multiple supports.
    # - High-precision product/tool anchors may auto-handle with one relevant
    #   historical match because the intent itself is deterministic.
    normal_evidence_ok = (
        top_similarity >= 0.45 and supporting_matches >= 2
    )
    anchored_evidence_ok = (
        is_high_precision and top_similarity >= 0.30 and supporting_matches >= 1
    )

    if not (normal_evidence_ok or anchored_evidence_ok):
        if is_high_precision:
            reasons.append(
                f"high-precision intent still has weak historical evidence "
                f"(top similarity {top_similarity:.2f}, supports {supporting_matches})"
            )
        else:
            reasons.append(
                f"historical evidence is not strong enough "
                f"(top similarity {top_similarity:.2f}, supports {supporting_matches})"
            )

    if reasons:
        return {
            "decision": "ESCALATE",
            "reason": "; ".join(reasons),
            "top_similarity": top_similarity,
            "supporting_matches": supporting_matches,
        }

    return {
        "decision": "AUTO-HANDLE",
        "reason": (
            "Intent confidence is high, the response generator supports "
            "auto-handling, and retrieval evidence meets the balanced "
            "evidence policy."
        ),
        "top_similarity": top_similarity,
        "supporting_matches": supporting_matches,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--message", required=True)
    parser.add_argument("--index-dir", default="data/retrieval_index")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured in this PowerShell session."
        )

    client = genai.Client(api_key=api_key)

    print("=" * 72)
    print("HIVER APPLESUPPORT AI AGENT")
    print("=" * 72)
    print(f"Customer: {args.message}")

    print("\n[1/4] Classifying intent...")
    try:
        intent_result = classify_intent(client, args.message)
    except GeminiQuotaError as exc:
        print("\n" + "=" * 72)
        print("GEMINI API LIMIT REACHED")
        print("=" * 72)
        print(str(exc))
        print("Please try again later.")
        return
    print(f"Intent: {intent_result['intent']}")
    print(f"Confidence: {intent_result['confidence']:.4f}")
    print(f"Reason: {intent_result['reason']}")

    print("\n[2/4] Retrieving historical cases...")
    vectorizer, matrix, metadata = load_retriever(args.index_dir)
    cases = retrieve(
        args.message,
        vectorizer,
        matrix,
        metadata,
        args.top_k,
    )
    print(f"Retrieved cases: {len(cases)}")

    for i, case in enumerate(cases, 1):
        print(
            f"  {i}. similarity={case['similarity']:.4f} "
            f"| {case['customer'][:110]}"
        )

    print("\n[3/4] Generating grounded reply...")
    try:
        generation_result = generate_grounded_reply(
            client, args.message, intent_result, cases
        )
    except GeminiQuotaError as exc:
        print("\n" + "=" * 72)
        print("GEMINI API LIMIT REACHED")
        print("=" * 72)
        print(str(exc))
        print("Please try again later.")
        return
    print(
        f"Generator recommendation: "
        f"{generation_result['model_recommendation']}"
    )
    print(
        f"Generator reason: "
        f"{generation_result['model_reason']}"
    )

    print("\n[4/4] Applying trust/escalation gate...")
    gate = trust_gate(intent_result, cases, generation_result)

    print("\n" + "=" * 72)
    print("FINAL AGENT OUTPUT")
    print("=" * 72)
    print(f"FINAL DECISION: {gate['decision']}")
    print(f"DECISION REASON: {gate['reason']}")
    print(f"TOP SIMILARITY: {gate['top_similarity']:.4f}")
    print(f"SUPPORTING MATCHES: {gate['supporting_matches']}")

    print("\nDRAFT REPLY:")
    print(generation_result["draft_reply"])

    print("\nEVIDENCE SUMMARY:")
    print(generation_result["evidence_summary"])


if __name__ == "__main__":
    main()
