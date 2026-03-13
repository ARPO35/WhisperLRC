# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

project_dir = Path(SPECPATH).resolve().parent
block_cipher = None


def _try_collect_submodules(package: str) -> list[str]:
    try:
        return collect_submodules(package)
    except Exception:
        return []


datas = [
    (str(project_dir / 'prompt.txt'), '.'),
    (str(project_dir / 'preferences.txt'), '.'),
    (str(project_dir / 'settings.toml.example'), '.'),
]
datas += collect_data_files('whisperlrc.review_server', includes=['static/*'])
datas += collect_data_files('faster_whisper')

binaries = []
for package_name in ('ctranslate2', 'tokenizers', 'av'):
    binaries += collect_dynamic_libs(package_name)

hiddenimports = []
for package_name in (
    'faster_whisper',
    'ctranslate2',
    'av',
    'uvicorn',
    'whisperlrc.review_server',
):
    hiddenimports += _try_collect_submodules(package_name)


a = Analysis(
    ['main.py'],
    pathex=[str(project_dir)],
    binaries=binaries,
    datas=datas,
    hiddenimports=sorted(set(hiddenimports)),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='WhisperLRC',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='WhisperLRC',
    contents_directory='.',
)
