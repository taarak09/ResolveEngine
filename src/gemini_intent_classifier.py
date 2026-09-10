
import argparse
import json
import os

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
        "Apple Watch, AirPods, headphones, or other Apple/device accessories.",
}


def build_prompt(message: str) -> str:
    intent_text = "\n".join(
        f"- {name}: {INTENT_DEFINITIONS[name]}"
        for name in INTENTS
    )

    return f"""
You are classifying a real AppleSupport customer-support message.

Customer message:
{message}

Choose exactly ONE intent from this list:

{intent_text}

Rules:
1. Choose the customer's primary issue, not merely a word mentioned in the message.
2. If an iOS update is explicitly the cause of a separate device problem (for example,
   battery drain, freezing, overheating, or loss of calls), prefer the specialized
   problem intent when it is clearly the primary issue. Use IOS_UPDATE_PROBLEM only
   when the update/install itself is the problem or the customer is asking about doing
   the update.
3. Use OTHER_OR_UNCLEAR when the message is only a vague follow-up such as "yes",
   "thanks", "I tried that", or otherwise lacks enough information.
4. Return JSON only, with exactly these fields:
{{
  "intent": "one intent from the list",
  "confidence": 0.0,
  "reason": "one concise sentence"
}}

The confidence must be a number from 0 to 1.
"""


def classify(message: str):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    client = genai.Client(api_key=api_key)

    interaction = client.interactions.create(
        model=MODEL_NAME,
        input=build_prompt(message),
    )

    raw = interaction.output_text.strip()

    # Be tolerant of accidental markdown code fences.
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    result = json.loads(raw)

    if result.get("intent") not in INTENTS:
        raise ValueError(
            f"Model returned unsupported intent: {result.get('intent')}"
        )

    confidence = float(result.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))

    return {
        "intent": result["intent"],
        "confidence": confidence,
        "reason": str(result.get("reason", "")).strip(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--message", required=True)
    args = parser.parse_args()

    print("=" * 70)
    print("GEMINI INTENT CLASSIFICATION")
    print("=" * 70)
    print(f"Message: {args.message}")

    result = classify(args.message)

    print(f"Intent: {result['intent']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Reason: {result['reason']}")


if __name__ == "__main__":
    main()
