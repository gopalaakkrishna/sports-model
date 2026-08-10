"""One-command Kalshi setup. Run this yourself; paste into YOUR terminal only.

    python src/setup_kalshi.py

It asks for your Key ID and your private key, stores the key in a file outside
this project, and sets the two environment variables the rest of the code reads.

Nothing is printed back, nothing is written into the project folder, and the key
never appears in a chat log or a commit. The private key is entered with echo
turned off, so it will not even appear on your screen.
"""

from __future__ import annotations

import getpass
import os
import subprocess
import sys
from pathlib import Path

KEY_DIR = Path.home() / ".kalshi"
KEY_FILE = KEY_DIR / "kalshi_key.pem"


def main() -> None:
    print("Kalshi setup\n" + "-" * 40)
    print("Nothing you type here is shown on screen or saved to this project.\n")

    key_id = input("1) Paste your API Key ID and press Enter:\n   ").strip()
    if not key_id:
        print("No Key ID entered — nothing changed.")
        return

    print("\n2) Your private key. Two options:")
    print("   a) type the PATH to the file Kalshi downloaded, or")
    print("   b) paste the key text itself (input stays hidden)")
    entry = input("   Path, or press Enter to paste instead:\n   ").strip().strip('"')

    if entry:
        src = Path(entry).expanduser()
        # A directory passes .exists() and then blows up on read_text(), so the
        # check has to be is_file(), not exists().
        if src.is_dir():
            print(f"   {src} is a folder, not a file.")
            cands = [p for p in src.iterdir()
                     if p.is_file() and p.suffix.lower() in (".pem", ".key", ".txt")]
            if cands:
                print("   Files in there that could be the key:")
                for c in cands[:10]:
                    print(f"     {c}")
                print("   Re-run and give the full path to one of those.")
            return
        if not src.is_file():
            print(f"   No file at {src} — nothing changed.")
            print("   Tip: include the filename, e.g. "
                  r"C:\Users\Gohan\Downloads\kalshi_key.pem")
            return
        try:
            key_text = src.read_text()
        except (PermissionError, OSError) as e:
            print(f"   Could not read {src}: {type(e).__name__}")
            return
    else:
        print("\n   Paste the key, then press Enter.")
        print("   (it will not appear as you type — that is expected)")
        key_text = getpass.getpass("   key: ")
        # Restore the PEM line breaks if pasting flattened them.
        if "-----BEGIN" in key_text and "\n" not in key_text.strip():
            body = key_text
            for marker in ("-----BEGIN RSA PRIVATE KEY-----",
                           "-----END RSA PRIVATE KEY-----",
                           "-----BEGIN PRIVATE KEY-----",
                           "-----END PRIVATE KEY-----"):
                body = body.replace(marker, f"\n{marker}\n")
            parts = [p.strip() for p in body.split("\n") if p.strip()]
            rebuilt = []
            for p in parts:
                if p.startswith("-----"):
                    rebuilt.append(p)
                else:
                    rebuilt.extend(p[i:i + 64] for i in range(0, len(p), 64))
            key_text = "\n".join(rebuilt) + "\n"

    if "PRIVATE KEY" not in key_text:
        print("   That does not look like a private key file — nothing changed.")
        return

    KEY_DIR.mkdir(parents=True, exist_ok=True)
    KEY_FILE.write_text(key_text)
    try:
        # Lock the file down to the current user only.
        subprocess.run(["icacls", str(KEY_FILE), "/inheritance:r",
                        "/grant:r", f"{os.environ.get('USERNAME', '')}:R"],
                       capture_output=True, check=False)
    except Exception:
        pass
    print(f"\n   Key saved to {KEY_FILE} (outside the project folder).")

    for name, val in (("KALSHI_KEY_ID", key_id),
                      ("KALSHI_PRIVATE_KEY_PATH", str(KEY_FILE))):
        subprocess.run(["setx", name, val], capture_output=True, check=False)
        os.environ[name] = val
    print("   Environment variables set.\n")

    print("Checking...")
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        import kalshi_auth
        kalshi_auth.check()
    except Exception as e:
        print(f"  check failed: {type(e).__name__}: {str(e)[:200]}")
        print("  The settings are saved. Open a NEW terminal and run:")
        print("    python src/kalshi_auth.py")
        return
    print("\nDone. New terminals will pick this up automatically.")


if __name__ == "__main__":
    main()
