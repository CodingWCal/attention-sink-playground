"""Curated preset prompts. Each illustrates a specific point from the PRD (§14).

Keep the dict ordered: the first entry is the default loaded on arrival.
"""

PRESETS: dict[str, str] = {
    "A normal sentence": "The quick brown fox jumps over the lazy dog.",
    "A repeated phrase": "I really really really really want to go home now.",
    "A very short prompt": "Hello there.",
    "A long prompt": (
        "In the early morning the city slowly wakes, traffic begins to hum "
        "along the avenues, vendors raise their shutters, and commuters spill "
        "out of the stations into the pale gold light of another working day."
    ),
    "Starts with punctuation": "— and then everything changed in an instant.",
    "A repeated prefix": "Paris Paris Paris is the capital of France.",
}

DEFAULT_LABEL = next(iter(PRESETS))
