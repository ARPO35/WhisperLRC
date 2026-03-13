from __future__ import annotations

import sys
from pathlib import Path

REQUIRED_PATHS = [
    'WhisperLRC.exe',
    'prompt.txt',
    'preferences.txt',
    'settings.toml.example',
    'whisperlrc/review_server/static/index.html',
]


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit('usage: verify_dist.py <dist_dir>')

    dist_dir = Path(argv[1]).resolve()
    missing = [rel for rel in REQUIRED_PATHS if not (dist_dir / rel).exists()]
    if missing:
        for rel in missing:
            print(f'MISSING: {rel}')
        return 1

    for rel in REQUIRED_PATHS:
        print(f'OK: {rel}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
