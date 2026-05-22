# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for Soccer Highlight GUI

import sys
from pathlib import Path

block_cipher = None

# ultralytics の yolo11n.pt を同梱する場合は datas に追加
# デフォルトは実行ファイルと同じフォルダに置く想定

a = Analysis(
    ['soccer_highlight_gui.py'],
    pathex=[],
    binaries=[],
    datas=[
        # YOLOモデルを同梱したい場合はコメントアウトを外す:
        # ('yolo11n.pt', '.'),
    ],
    hiddenimports=[
        # PyQt6
        'PyQt6.QtWidgets',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        # numpy / scipy / librosa 関連
        'numpy',
        'librosa',
        'librosa.core',
        'librosa.feature',
        'scipy',
        'scipy.signal',
        'soundfile',
        'audioread',
        'resampy',
        'numba',
        # ultralytics
        'ultralytics',
        'ultralytics.models',
        'ultralytics.nn',
        # opencv
        'cv2',
        # その他
        'PIL',
        'yaml',
        'packaging',
        'psutil',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'IPython',
        'jupyter',
        'notebook',
        'tkinter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher
)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SoccerHighlight',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # GUIアプリなのでコンソール非表示
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon='icon.ico',       # アイコンを用意する場合はコメントアウトを外す
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SoccerHighlight',
)
