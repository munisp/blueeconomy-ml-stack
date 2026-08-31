"""RFC 8785 (JCS) JSON canonicalization.

Implements exactly the rules required by the platform envelope-signature
scheme (blueeconomy-contracts/docs/envelope-signature.md):

- Object members sorted by key using UTF-16 code-unit order; no whitespace.
- Strings use minimal JSON escaping; non-ASCII emitted raw (UTF-8), never
  ``\\u``-escaped except the mandatory control-character escapes.
- Numbers follow ECMAScript ``Number::toString`` semantics (shortest
  round-trip; exponential form only below 1e-6 or at/above 1e21).
- No duplicate object keys (rejected on parse by the caller).
"""

from __future__ import annotations

import math
from typing import Any

_ESCAPES = {
    '"': '\\"',
    "\\": "\\\\",
    "\b": "\\b",
    "\f": "\\f",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}


def _escape_string(s: str) -> str:
    out = []
    for ch in s:
        esc = _ESCAPES.get(ch)
        if esc is not None:
            out.append(esc)
        elif ord(ch) < 0x20:
            out.append("\\u%04x" % ord(ch))
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def _es_number_digits(value: float) -> tuple[str, int]:
    """Return (shortest digit string ``s``, decimal exponent ``n``) such that
    ``int(s) * 10**(n - len(s)) == value``, matching ECMAScript's choice of
    the shortest string that round-trips."""
    repr_s = repr(value)
    if "e" in repr_s or "E" in repr_s:
        mantissa, _, exp = repr_s.lower().partition("e")
        exp10 = int(exp)
    else:
        mantissa, exp10 = repr_s, 0
    if "." in mantissa:
        int_part, _, frac_part = mantissa.partition(".")
    else:
        int_part, frac_part = mantissa, ""
    digits = int_part + frac_part
    # value = D * 10^(n - k) where D = int(digits stripped of leading zeros),
    # k = number of significant digits.
    stripped = digits.lstrip("0") or "0"
    # position of decimal point relative to first significant digit
    lead_zeros = len(digits) - len(stripped)
    point_index = len(int_part)  # digits[0:point_index] is the integer part
    n = point_index - lead_zeros + exp10
    # strip trailing zeros (shortest form never needs them)
    stripped = stripped.rstrip("0") or "0"
    return stripped, n


def _format_number(value: int | float) -> str:
    if isinstance(value, bool):
        raise TypeError("bool is not a JCS number")
    if isinstance(value, int):
        if abs(value) >= 2**53:
            raise ValueError("integer outside IEEE-754 safe range")
        return str(value)
    f = float(value)
    if math.isnan(f) or math.isinf(f):
        raise ValueError("non-finite number cannot be canonicalized")
    if f == 0:
        return "0"
    if f.is_integer() and abs(f) < 1e21:
        if abs(f) >= 2**53:
            raise ValueError("number outside IEEE-754 safe range")
        return str(int(f))
    sign = "-" if f < 0 else ""
    digits, n = _es_number_digits(abs(f))
    k = len(digits)
    if k <= n <= 21:
        return sign + digits + "0" * (n - k)
    if 0 < n <= 21:
        return sign + digits[:n] + "." + digits[n:]
    if -6 < n <= 0:
        return sign + "0." + "0" * (-n) + digits
    # Exponential notation.
    if k == 1:
        mantissa = digits
    else:
        mantissa = digits[0] + "." + digits[1:]
    exp = n - 1
    return sign + mantissa + "e" + ("+" if exp >= 0 else "-") + str(abs(exp))


def _serialize(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return _escape_string(value)
    if isinstance(value, (int, float)):
        return _format_number(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_serialize(v) for v in value) + "]"
    if isinstance(value, dict):
        # Sort by UTF-16 code-unit order: encoding to UTF-16-BE gives a byte
        # sequence whose lexicographic order matches code-unit order.
        items = sorted(value.items(), key=lambda kv: kv[0].encode("utf-16-be"))
        return "{" + ",".join(
            _escape_string(k) + ":" + _serialize(v) for k, v in items
        ) + "}"
    raise TypeError(f"unsupported type for JCS: {type(value)!r}")


def canonicalize(value: Any) -> bytes:
    """Return the RFC 8785 canonical JSON of ``value`` as UTF-8 bytes."""
    return _serialize(value).encode("utf-8")


def canonicalize_str(value: Any) -> str:
    """Return the RFC 8785 canonical JSON of ``value`` as a string."""
    return _serialize(value)
