"""Kalshi authenticated access — credentials read from the environment ONLY.

NEVER paste an API key or private key into a chat, a commit, or a source file.
Kalshi credentials can place and cancel orders, not just read data, so a leaked
key is a financial risk rather than a privacy one.

Setup (do this yourself; nothing here needs to be shown to anyone):

  1. Kalshi gives you an API Key ID and a downloaded RSA private key file.
  2. Put the private key file somewhere OUTSIDE this repository, e.g.
         C:\\Users\\<you>\\.kalshi\\kalshi_private_key.pem
  3. Set two environment variables (PowerShell, permanent for your user):
         setx KALSHI_KEY_ID "your-key-id-here"
         setx KALSHI_PRIVATE_KEY_PATH "C:\\Users\\<you>\\.kalshi\\kalshi_private_key.pem"
     Open a new terminal afterwards so they take effect.

This module reads those variables at run time. It never prints, logs or writes
the key, and the repository never contains it.

SCOPE: this file signs READ requests only — balance, positions, fills, resting
orders. It deliberately contains no order-placement function. Trading decisions
and executions stay with you.
"""

from __future__ import annotations

import base64
import datetime as _dt
import os
import time
from pathlib import Path

import requests

BASE = "https://api.elections.kalshi.com/trade-api/v2"


class CredentialsMissing(RuntimeError):
    pass


def _load_private_key():
    """Import lazily so the rest of the project runs without `cryptography`."""
    try:
        from cryptography.hazmat.primitives import serialization
    except ImportError as e:
        raise CredentialsMissing(
            "the `cryptography` package is required: pip install cryptography"
        ) from e

    path = os.environ.get("KALSHI_PRIVATE_KEY_PATH")
    if not path:
        raise CredentialsMissing("KALSHI_PRIVATE_KEY_PATH is not set")
    p = Path(path)
    if not p.exists():
        raise CredentialsMissing(f"private key file not found at {p}")
    with p.open("rb") as fh:
        return serialization.load_pem_private_key(fh.read(), password=None)


def _sign(private_key, message: str) -> str:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding

    sig = private_key.sign(
        message.encode("utf-8"),
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode("utf-8")


def auth_headers(method: str, path: str) -> dict:
    key_id = os.environ.get("KALSHI_KEY_ID")
    if not key_id:
        raise CredentialsMissing("KALSHI_KEY_ID is not set")
    pk = _load_private_key()
    ts = str(int(time.time() * 1000))
    # Kalshi signs timestamp + METHOD + path (path without query string).
    msg = ts + method.upper() + path
    return {
        "KALSHI-ACCESS-KEY": key_id,
        "KALSHI-ACCESS-SIGNATURE": _sign(pk, msg),
        "KALSHI-ACCESS-TIMESTAMP": ts,
        "Content-Type": "application/json",
    }


def get(path: str, params: dict | None = None) -> dict:
    """Authenticated GET. `path` starts with /trade-api/v2/..."""
    headers = auth_headers("GET", path)
    r = requests.get("https://api.elections.kalshi.com" + path,
                     headers=headers, params=params, timeout=45)
    r.raise_for_status()
    return r.json()


def check() -> None:
    """Verify credentials work, without ever displaying them."""
    try:
        bal = get("/trade-api/v2/portfolio/balance")
    except CredentialsMissing as e:
        print(f"not configured: {e}")
        print("See the setup notes at the top of this file.")
        return
    except requests.HTTPError as e:
        print(f"credentials present but rejected: HTTP {e.response.status_code}")
        print(f"  {e.response.text[:200]}")
        return
    # Print only the derived figure, never the credential.
    cents = bal.get("balance")
    print("credentials OK")
    if isinstance(cents, (int, float)):
        print(f"  account balance: ${cents / 100:,.2f}")

    fills = get("/trade-api/v2/portfolio/fills", {"limit": 5})
    n = len(fills.get("fills", []))
    print(f"  recent fills readable: {n}")


if __name__ == "__main__":
    check()
