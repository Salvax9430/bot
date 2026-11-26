from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class KeyEvent:
    """Represents a single key action and delay before/after execution."""

    kind: str  # "char" | "backspace" | "pause"
    value: str
    delay_sec: float


_COMMON_DIGRAPHS = frozenset(
    {
        "th", "he", "in", "er", "an", "on", "re", "ed", "nd", "ha",
        "at", "en", "es", "of", "or", "nt", "ea", "ti", "to", "it",
        "st", "io", "le", "is", "ou", "ar", "as", "de", "rt", "ve",
    }
)

_ADJACENT_KEYS: dict[str, str] = {
    "a": "sqwz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wrsdf",
    "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "i": "ujko", "j": "huikmn",
    "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm", "o": "iklp",
    "p": "ol", "q": "wa", "r": "edft", "s": "awedxz", "t": "rfgy",
    "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tugh",
    "z": "asx",
}


class TypingModel:
    """Generates human-like keystroke timing and occasional typo/correction sequences."""

    def __init__(
        self,
        wpm: float = 68.0,
        sigma: float = 0.24,
        typo_prob: float = 0.025,
        fatigue_per_minute: float = 0.015,
        max_fatigue_multiplier: float = 1.35,
        rng: random.Random | None = None,
    ) -> None:
        # Average word length ~5 chars -> chars/sec = (wpm * 5) / 60
        chars_per_sec = max(1.0, (max(10.0, float(wpm)) * 5.0) / 60.0)
        self.base_iki_sec = 1.0 / chars_per_sec
        self.sigma = max(0.05, float(sigma))
        self.typo_prob = max(0.0, min(0.25, float(typo_prob)))
        self.fatigue_per_minute = max(0.0, float(fatigue_per_minute))
        self.max_fatigue_multiplier = max(1.0, float(max_fatigue_multiplier))
        self.rng = rng if rng is not None else random.Random()
        self.active_seconds: float = 0.0

    def _fatigue_multiplier(self) -> float:
        minutes = self.active_seconds / 60.0
        return min(self.max_fatigue_multiplier, 1.0 + self.fatigue_per_minute * minutes)

    def _sample_lognormal_delay(self, multiplier: float = 1.0, extra_pause: float = 0.0) -> float:
        mean_sec = max(0.012, self.base_iki_sec * multiplier * self._fatigue_multiplier())
        mu = math.log(mean_sec)
        raw = self.rng.lognormvariate(mu, self.sigma)
        # Clamp extreme outliers so typing stays natural and responsive
        clamped = max(0.015, min(raw, mean_sec * 3.2)) + max(0.0, extra_pause)
        self.active_seconds += clamped
        return round(clamped, 4)

    def _neighbor_char(self, ch: str) -> str:
        low = ch.lower()
        neighbors = _ADJACENT_KEYS.get(low)
        if not neighbors:
            return ch
        picked = self.rng.choice(neighbors)
        return picked.upper() if ch.isupper() else picked

    def plan_text(self, text: str, allow_typos: bool = True) -> list[KeyEvent]:
        """Convert a string into timed KeyEvent items including optional typo + fix."""
        events: list[KeyEvent] = []
        n = len(text)

        for idx, ch in enumerate(text):
            mult = 1.0
            extra_pause = 0.0

            if idx > 0:
                pair = (text[idx - 1] + ch).lower()
                if pair in _COMMON_DIGRAPHS:
                    mult *= 0.84

            if ch in ".!?":
                extra_pause += self.rng.uniform(0.14, 0.34)
            elif ch in ",;:":
                extra_pause += self.rng.uniform(0.06, 0.16)
            elif ch == "\n":
                extra_pause += self.rng.uniform(0.12, 0.28)
            elif ch == " ":
                mult *= 0.95
            elif ch.isupper() or not ch.isalnum():
                mult *= 1.15

            can_typo = (
                allow_typos
                and self.typo_prob > 0.0
                and ch.isalpha()
                and ch.lower() in _ADJACENT_KEYS
                and idx < n - 1
            )

            if can_typo and self.rng.random() < self.typo_prob:
                wrong_ch = self._neighbor_char(ch)
                events.append(
                    KeyEvent(
                        kind="char",
                        value=wrong_ch,
                        delay_sec=self._sample_lognormal_delay(mult, extra_pause),
                    )
                )
                # Brief realization pause, then backspace, then retype intended character
                events.append(
                    KeyEvent(
                        kind="pause",
                        value="",
                        delay_sec=self._sample_lognormal_delay(1.0, self.rng.uniform(0.08, 0.22)),
                    )
                )
                events.append(
                    KeyEvent(
                        kind="backspace",
                        value="\b",
                        delay_sec=self._sample_lognormal_delay(1.05, 0.02),
                    )
                )

            events.append(
                KeyEvent(
                    kind="char",
                    value=ch,
                    delay_sec=self._sample_lognormal_delay(mult, extra_pause if not can_typo else 0.0),
                )
            )

        return events