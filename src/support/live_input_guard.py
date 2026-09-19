"""Pure owner-scoped input checks; the caller owns locking and persistence."""
from __future__ import annotations

import hashlib
from functools import lru_cache
import json
import math
from pathlib import Path
import re
import string
import unicodedata

MAX_ATTEMPTS = 3
LOCK_SECONDS = 300
MAX_RECEIPTS = 64
_KEY_ROWS = ("йцукенгшщзхъ", "фывапролджэ", "ячсмитьбю")
_KEY_POSITIONS = {char: (column + row * 0.35, row)
                  for row, keys in enumerate(_KEY_ROWS) for column, char in enumerate(keys)}
_VOWELS = frozenset("аеёиоуыэюя")


@lru_cache(maxsize=1)
def _spelling_reference():
    reference = json.loads(Path(__file__).with_name("live_input_reference.json").read_text(encoding="utf-8"))
    if reference["version"] != 1:
        raise ValueError("Unsupported live input reference")
    return frozenset(reference["words"]), frozenset(reference["triples"])


def _irregular_keyboard_word(word: str, *, minimum_length: int = 8) -> bool:
    # Vocabulary absence alone cannot reject names, technical terms or typos.
    # Require three signals: unfamiliar triples, keyboard concentration and
    # unusual consonant order (or sustained low-diversity repetition).
    if len(word) < minimum_length or not re.fullmatch(r"[а-яё]+", word):
        return False
    known_words, triples = _spelling_reference()
    if word in known_words:
        return False
    unseen = [word[index:index + 3] for index in range(len(word) - 2)
              if word[index:index + 3] not in triples]
    if len(unseen) / (len(word) - 2) < 0.35:
        return False
    consonants = re.findall(r"[^аеёиоуыэюя]+", word)
    long_run = max(map(len, consonants), default=0) >= 6
    row_fraction = max(sum(char in row for char in word) for row in _KEY_ROWS) / len(word)
    adjacent = 0
    for first, second in zip(word, word[1:]):
        if first in _KEY_POSITIONS and second in _KEY_POSITIONS:
            x1, y1 = _KEY_POSITIONS[first]
            x2, y2 = _KEY_POSITIONS[second]
            adjacent += abs(x1 - x2) <= 1.5 and abs(y1 - y2) <= 1
    if row_fraction < 0.75 and adjacent / (len(word) - 1) < 0.70 and not long_run:
        return False
    unusual_cluster = any(not set(triple) & _VOWELS for triple in unseen)
    repeated = len(word) >= 10 and len(word) >= 2.5 * len(set(word))
    # A home-row smash may contain many vowels and no long consonant run.
    # This stronger combination is independent of the consonant-order signal.
    concentrated_noise = row_fraction >= 0.95 and len(unseen) / (len(word) - 2) >= 0.65
    return unusual_cluster or repeated or concentrated_noise


def inspect_message(message: str) -> str | None:
    """Conservative noise check; unfamiliar vocabulary alone is not a strike."""
    text = message.strip().casefold()
    if not text:
        return "empty"
    if all(char.isspace() or char in string.punctuation
           or unicodedata.category(char).startswith("P") for char in text):
        return "punctuation_only"
    # Ignore separators for noise detection, but preserve numeric/alphanumeric
    # identifiers and other symbols. Unknown vocabulary is not evidence of noise.
    compact = "".join(char for char in text if not (
        char.isspace() or char in string.punctuation
        or unicodedata.category(char).startswith("P")))
    if compact.isalpha():
        if len(compact) >= 8 and len(set(compact)) == 1:
            return "repeated_letter"
        # Restrict irregular smashes to tiny key clusters with high repetition.
        # Do not use the full home row: it also spells ordinary Russian words.
        letters = set(compact)
        if len(compact) >= 8 and len(compact) >= 3 * len(letters):
            if any(letters <= set(cluster) for cluster in ("фывп", "asdf")):
                return "keyboard_noise"
        # Irregular Latin home-cluster smashes need not repeat a whole block.
        # Require a tiny cluster, repeated physical-key triples and a long
        # consonant run together; unknown English words alone are not noise.
        if (len(compact) >= 8 and letters <= set("asdfg")
                and any(compact.count(block) >= 2 for block in ("asd", "sdf", "dfg"))
                and re.search(r"[sdfg]{4,}", compact)):
            return "keyboard_noise"
        for width in range(2, 7):
            if len(compact) >= max(12, width * 3) and len(compact) % width == 0:
                if compact == compact[:width] * (len(compact) // width):
                    return "repeated_pattern"
        # Two complete keyboard-row cycles are already high-confidence noise.
        for row in ("йцукен", "qwerty"):
            if len(compact) >= 2 * len(row) and compact == row * (len(compact) // len(row)):
                return "keyboard_noise"
        # Two short keyboard blocks were missed by the >=12/three-cycle rule.
        # Require one physical row so ordinary repeated word fragments alone
        # do not trigger this shorter check. Identifiers with digits are exempt.
        for row in (*_KEY_ROWS, "qwertyuiop", "asdfghjkl", "zxcvbnm"):
            if len(compact) < 8 or not set(compact) <= set(row):
                continue
            for width in range(2, 7):
                if len(compact) >= width * 2 and len(compact) % width == 0:
                    if compact == compact[:width] * (len(compact) // width):
                        # Latin home-row words such as "glassglass" also repeat.
                        # Require an actual consecutive-key triple for Latin.
                        if (not compact.isascii() or any(
                                row[index:index + 3] in compact
                                or row[index:index + 3][::-1] in compact
                                for index in range(len(row) - 2))):
                            return "keyboard_blocks"
            if re.fullmatch(r"[a-z]+", compact):
                adjacent = sum(abs(row.index(a) - row.index(b)) <= 1
                               for a, b in zip(compact, compact[1:]))
                if adjacent / (len(compact) - 1) >= 0.85:
                    return "keyboard_walk"
        # Keep word boundaries: a real support sentence containing an unknown
        # module name is useful. This gate targets messages consisting of noise,
        # not every unfamiliar word embedded in an otherwise meaningful request.
        words = re.findall(r"[^\W\d_]+", text)
        # Splitting a short smash into two fragments must not change its score.
        # Keep this restricted to short words, and retain the same strong
        # statistical check; ordinary sentences still use word boundaries.
        if len(words) > 1 and all(4 <= len(word) < 8 for word in words):
            if _irregular_keyboard_word(compact):
                return "irregular_keyboard_noise"
        if len(compact) >= 8 and words and all(
                _irregular_keyboard_word(word, minimum_length=4) for word in words):
            return "irregular_keyboard_noise"
    return None


def _decision(*, allowed: bool, attempts: int, code: str, message: str,
              blocked_until: float | None = None, now: float = 0) -> dict:
    return {
        "allowed": allowed,
        "blocked": blocked_until is not None,
        "retry_after": max(0, math.ceil(blocked_until - now)) if blocked_until is not None else 0,
        "blocked_until": blocked_until,
        "attempts_remaining": max(0, MAX_ATTEMPTS - attempts),
        "message": message,
        "code": code,
    }


def check_input_status(state: dict, now: float) -> dict:
    """Inspect the lock without consuming a retry; reset only an expired lock."""
    if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now):
        raise ValueError("now must be a finite timestamp")
    until = state.get("blocked_until")
    if until is not None:
        if now < until:
            return _decision(allowed=False, attempts=MAX_ATTEMPTS, code="input_blocked",
                             message="Отправка временно недоступна. Попробуйте через пять минут после начала блокировки.",
                             blocked_until=until, now=now)
        state.update(attempts=0, blocked_until=None, receipts={})
    return _decision(allowed=True, attempts=state.get("attempts", 0), code="allowed", message="")


def evaluate_input(state: dict, message: str, key: str, now: float) -> dict:
    """Mutate a dedicated persisted owner dict and return a detached decision.

    ``now`` is a finite Unix timestamp in seconds. Active locks take precedence
    over retries and never extend. Expiry clears the strike/retry epoch. Within
    an epoch the last 64 keys bind to exact payloads; retained retries cannot
    count or reset a strike twice. Only hashes are retained, not message text.
    Call under the same owner lock as persistence and downstream request setup.
    """
    if not isinstance(message, str) or not isinstance(key, str) or not key:
        raise ValueError("message and a nonempty idempotency key must be strings")
    status = check_input_status(state, now)
    if status["blocked"]:
        return status

    attempts = state.get("attempts", 0)
    receipts = state.setdefault("receipts", {})
    payload_hash = hashlib.sha256(message.encode("utf-8")).hexdigest()
    previous = receipts.get(key)
    if previous is not None:
        if previous["payload_sha256"] != payload_hash:
            return _decision(allowed=False, attempts=attempts, code="idempotency_conflict",
                             message="Этот запрос уже использован для другого сообщения. Отправьте новое сообщение.")
        return dict(previous["decision"])

    reason = inspect_message(message)
    if reason is None:
        state.update(attempts=0, blocked_until=None)
        result = _decision(allowed=True, attempts=0, code="allowed", message="")
    else:
        attempts += 1
        state["attempts"] = attempts
        if attempts >= MAX_ATTEMPTS:
            until = now + LOCK_SECONDS
            state["blocked_until"] = until
            result = _decision(allowed=False, attempts=attempts, code="input_blocked",
                               message="Не удалось понять три сообщения подряд. Отправка приостановлена на пять минут; затем опишите вопрос словами.",
                               blocked_until=until, now=now)
        else:
            remaining = MAX_ATTEMPTS - attempts
            result = _decision(allowed=False, attempts=attempts, code="invalid_message",
                               message=f"Не удалось понять сообщение. Опишите вопрос словами, например: «нет сети». До временной блокировки осталось попыток: {remaining}.")
    receipts[key] = {"payload_sha256": payload_hash, "decision": dict(result)}
    while len(receipts) > MAX_RECEIPTS:
        del receipts[next(iter(receipts))]
    return result
