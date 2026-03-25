"""Linguistic register extractor: slang density, message length, punctuation, formality."""

from __future__ import annotations

import re
import time

from ghost_sense.extractor.base import BaseExtractor
from ghost_sense.models import SignalEvent, SignalType

# Slang/informal lexicon — seed with common internet/casual terms
# This gets extended over time as patterns emerge
SLANG_TOKENS = {
    "lol", "lmao", "lmfao", "rofl", "bruh", "bro", "nah", "yea", "yeah", "yep",
    "nope", "gonna", "wanna", "gotta", "kinda", "sorta", "idk", "imo", "imho",
    "tbh", "tbf", "ngl", "fr", "frfr", "smh", "fwiw", "iirc", "afaik", "afk",
    "btw", "omg", "omfg", "wtf", "wth", "stfu", "sus", "lowkey", "highkey",
    "deadass", "fam", "vibe", "vibes", "vibing", "bet", "cap", "nocap", "no cap",
    "ong", "slay", "fire", "mid", "bussin", "goated", "based", "cringe",
    "yolo", "fomo", "tho", "doe", "rn", "irl", "haha", "heh", "lel", "kek",
    "pog", "poggers", "copium", "hopium", "ratio", "w", "l", "dw", "np",
    "ty", "thx", "pls", "plz", "u", "ur", "r", "k", "ok", "okie",
    "aight", "ight", "fs", "ofc", "obvi", "prolly", "p", "v",  # "p sure", "v cool"
    "yo", "ay", "ayy", "ayo", "mans", "dawg", "homie", "lit",
    "chill", "chillin", "tryna", "finna", "boutta", "ion",  # "I don't"
    "mf", "tf", "istg", "icl", "wym", "hbu", "wbu",
}

EMOJI_PATTERN = re.compile(
    "[\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"  # supplemental
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "]+",
    flags=re.UNICODE,
)

EMOTICON_PATTERN = re.compile(
    r"(?:[:;][-']?[)(DPpOo3><|/\\])|(?:[)(DPp][-']?[:;])|<3|</3|\bxD\b|\bXD\b|:'\("
)


class RegisterExtractor(BaseExtractor):
    """Extracts per-message linguistic register features."""

    def extract(self, message: str, metadata: dict) -> list[SignalEvent]:
        ts = metadata.get("timestamp", time.time())
        events: list[SignalEvent] = []

        words = message.split()
        if not words:
            return events

        word_count = len(words)
        char_count = sum(len(w) for w in words)

        # --- Average word length ---
        avg_word_len = char_count / word_count
        events.append(SignalEvent(
            signal_type=SignalType.REGISTER,
            dimension="register.avg_word_length",
            value=avg_word_len,
            confidence=0.9,
            source_text=message[:120],
            timestamp=ts,
        ))

        # --- Slang ratio ---
        lower_words = [w.lower().strip(".,!?;:'\"()[]{}") for w in words]
        slang_count = sum(1 for w in lower_words if w in SLANG_TOKENS)
        slang_ratio = slang_count / word_count
        events.append(SignalEvent(
            signal_type=SignalType.REGISTER,
            dimension="register.slang_ratio",
            value=slang_ratio,
            confidence=0.85,
            source_text=message[:120],
            timestamp=ts,
        ))

        # --- Sentence count ---
        sentences = re.split(r'[.!?]+', message)
        sentence_count = len([s for s in sentences if s.strip()])
        events.append(SignalEvent(
            signal_type=SignalType.REGISTER,
            dimension="register.sentence_count",
            value=float(sentence_count),
            confidence=0.9,
            source_text=message[:120],
            timestamp=ts,
        ))

        # --- Punctuation density ---
        punct_chars = sum(1 for c in message if c in '.,!?;:\'"-()[]{}…')
        punct_density = punct_chars / len(message) if message else 0.0
        events.append(SignalEvent(
            signal_type=SignalType.REGISTER,
            dimension="register.punctuation_density",
            value=punct_density,
            confidence=0.9,
            source_text=message[:120],
            timestamp=ts,
        ))

        # --- Capitalization ratio ---
        alpha_chars = [c for c in message if c.isalpha()]
        cap_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars) if alpha_chars else 0.0
        events.append(SignalEvent(
            signal_type=SignalType.REGISTER,
            dimension="register.capitalization_ratio",
            value=cap_ratio,
            confidence=0.9,
            source_text=message[:120],
            timestamp=ts,
        ))

        # --- Emoji/emoticon density ---
        emoji_count = len(EMOJI_PATTERN.findall(message))
        emoticon_count = len(EMOTICON_PATTERN.findall(message))
        emoji_density = (emoji_count + emoticon_count) / word_count
        events.append(SignalEvent(
            signal_type=SignalType.REGISTER,
            dimension="register.emoji_density",
            value=emoji_density,
            confidence=0.85,
            source_text=message[:120],
            timestamp=ts,
        ))

        # --- Composite formality score ---
        # 0.0 = maximally informal, 1.0 = maximally formal
        # Weighted blend of inverse slang, word length, punctuation, capitalization
        formality = (
            (1.0 - slang_ratio) * 0.35
            + min(avg_word_len / 8.0, 1.0) * 0.25  # longer words = more formal
            + punct_density * 3.0 * 0.15  # more punctuation = more formal (scaled)
            + (cap_ratio * 5.0 if cap_ratio < 0.3 else 0.3) * 0.15  # some caps = formal, ALL CAPS = not
            + (1.0 - emoji_density) * 0.10
        )
        formality = max(0.0, min(1.0, formality))
        events.append(SignalEvent(
            signal_type=SignalType.REGISTER,
            dimension="register.formality_score",
            value=formality,
            confidence=0.8,
            source_text=message[:120],
            timestamp=ts,
        ))

        return events
