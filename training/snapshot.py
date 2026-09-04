"""Freeze the current agent into a self-contained opponent for A/B measurement.

    uv run python training/snapshot.py base_1773

Copies agent.py, evalnet.py, features.py and weights/net.pt into opponents/<name>/ as a
Python PACKAGE (with __init__.py) and rewrites the copied agent's imports to be relative. That
matters because training/elo.py loads every player into ONE process: two players that both did
`import evalnet` would share a single top-level module and a single weights file, so an old
snapshot and the live candidate would silently evaluate with the SAME net. As a package the
snapshot's imports resolve to opponents/<name>/evalnet.py and its own weights, so a frozen net
really is frozen -- which is the whole point of a baseline.

Only agent.py needs rewriting: evalnet.py and features.py have no in-repo imports, and
evalnet loads its weights from `Path(__file__).parent / weights`, which becomes the snapshot's
own copy automatically.
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def rewrite_agent_imports(text: str) -> str:
    """Make the copied agent import its sibling modules, not the top-level ones."""
    text = re.sub(r"^import evalnet$", "from . import evalnet", text, flags=re.MULTILINE)
    text = re.sub(r"^from features import", "from .features import", text, flags=re.MULTILINE)
    text = re.sub(r"^from evalnet import", "from .evalnet import", text, flags=re.MULTILINE)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the current agent as an opponent.")
    parser.add_argument("name", help="snapshot name, e.g. base_1773")
    parser.add_argument("--force", action="store_true", help="overwrite an existing snapshot")
    args = parser.parse_args()

    dest = ROOT / "opponents" / args.name
    if dest.exists():
        if not args.force:
            print(f"{dest} already exists; pass --force to overwrite")
            sys.exit(1)
        shutil.rmtree(dest)
    (dest / "weights").mkdir(parents=True)

    (dest / "__init__.py").write_text("", encoding="utf-8")
    agent_src = (ROOT / "agent.py").read_text(encoding="utf-8")
    (dest / "agent.py").write_text(rewrite_agent_imports(agent_src), encoding="utf-8")
    shutil.copy(ROOT / "evalnet.py", dest / "evalnet.py")
    shutil.copy(ROOT / "features.py", dest / "features.py")
    shutil.copy(ROOT / "weights" / "net.pt", dest / "weights" / "net.pt")

    print(f"froze current agent into {dest} (agent.py + evalnet.py + features.py + weights)")


if __name__ == "__main__":
    main()
