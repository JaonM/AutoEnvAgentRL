"""Detached Ed25519 attestations for portable material bundle manifests."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
from typing import Any


SCHEME = "ed25519-openssl-pkeyutl"


def _run(arguments: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
    )


def public_key_identity(public_key: Path) -> str:
    result = _run([
        "openssl", "pkey", "-pubin", "-in", str(public_key.resolve()),
        "-outform", "DER",
    ])
    if result.returncode != 0 or not result.stdout:
        raise ValueError("trusted public key is not a readable OpenSSL public key")
    return hashlib.sha256(result.stdout).hexdigest()


def private_key_public_identity(private_key: Path) -> str:
    result = _run([
        "openssl", "pkey", "-in", str(private_key.resolve()),
        "-pubout", "-outform", "DER",
    ])
    if result.returncode != 0 or not result.stdout:
        raise ValueError("signing private key is not a readable OpenSSL private key")
    return hashlib.sha256(result.stdout).hexdigest()


def signed_metadata(private_key: Path, public_key: Path, signature_file: str) -> dict[str, Any]:
    private_identity = private_key_public_identity(private_key)
    public_identity = public_key_identity(public_key)
    if private_identity != public_identity:
        raise ValueError("signing private key does not match trusted public key")
    return {
        "version": "1.0",
        "status": "signed",
        "scheme": SCHEME,
        "key_identity_sha256": public_identity,
        "signature_file": signature_file,
    }


def sign_file(payload: Path, signature: Path, private_key: Path) -> None:
    result = _run([
        "openssl", "pkeyutl", "-sign", "-inkey", str(private_key.resolve()),
        "-rawin", "-in", str(payload.resolve()), "-out", str(signature.resolve()),
    ])
    if result.returncode != 0 or not signature.is_file() or signature.stat().st_size == 0:
        signature.unlink(missing_ok=True)
        raise ValueError("OpenSSL failed to sign the bundle manifest")


def verify_file(
    payload: Path, signature: Path, public_key: Path, metadata: Any
) -> bool:
    if not isinstance(metadata, dict):
        return False
    try:
        identity = public_key_identity(public_key)
    except (OSError, ValueError):
        return False
    if metadata != {
        "version": "1.0",
        "status": "signed",
        "scheme": SCHEME,
        "key_identity_sha256": identity,
        "signature_file": signature.name,
    }:
        return False
    result = _run([
        "openssl", "pkeyutl", "-verify", "-pubin", "-inkey",
        str(public_key.resolve()), "-rawin", "-in", str(payload.resolve()),
        "-sigfile", str(signature.resolve()),
    ])
    return result.returncode == 0
