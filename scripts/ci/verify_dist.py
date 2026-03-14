from __future__ import annotations

import sys
from pathlib import Path

REQUIRED_FILES = [
    ('WhisperLRC.exe', ['WhisperLRC.exe']),
    ('prompt.txt', ['prompt.txt']),
    ('preferences.txt', ['preferences.txt']),
    ('settings.toml', ['settings.toml']),
    (
        'whisperlrc/review_server/static/index.html',
        [
            'whisperlrc/review_server/static/index.html',
            '_internal/whisperlrc/review_server/static/index.html',
        ],
    ),
]

FORBIDDEN_FILES = [
    'settings.toml.example',
]


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit('usage: verify_dist.py <dist_dir>')

    dist_dir = Path(argv[1]).resolve()
    missing: list[str] = []
    for label, candidates in REQUIRED_FILES:
        if not any((dist_dir / rel).exists() for rel in candidates):
            missing.append(label)

    if missing:
        for rel in missing:
            print(f'MISSING: {rel}')
        return 1

    forbidden = [rel for rel in FORBIDDEN_FILES if (dist_dir / rel).exists()]
    if forbidden:
        for rel in forbidden:
            print(f'FORBIDDEN: {rel}')
        return 1

    for label, candidates in REQUIRED_FILES:
        found = next(rel for rel in candidates if (dist_dir / rel).exists())
        print(f'OK: {label} -> {found}')
    for rel in FORBIDDEN_FILES:
        print(f'OK: absent -> {rel}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
