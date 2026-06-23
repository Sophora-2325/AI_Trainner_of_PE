# -*- mode: python ; coding: utf-8 -*-
# PyInstaller 打包配置
# 用法: pyinstaller fitness_coach.spec

import os
from PyInstaller.utils.hooks import collect_all, collect_data_files

block_cipher = None
root = os.path.abspath('.')

# mediapipe / cv2 需要额外收集数据文件
mp_datas, mp_binaries, mp_hidden = collect_all('mediapipe')
cv2_datas = collect_data_files('cv2')

datas = [
    ('config', 'config'),
    ('templates', 'templates'),
    ('web', 'web'),
    ('sports_twin.html', '.'),
    ('models/pose_landmarker_lite.task', 'models'),
    ('models/pose_landmarker_full.task', 'models'),
    ('models/pose_landmarker_heavy.task', 'models'),
    ('src', 'src'),
    ('scripts', 'scripts'),
] + mp_datas + cv2_datas

hiddenimports = mp_hidden + [
    'app',
    'web_coach',
    'scripts.auto_demo',
    'scripts.squat_keyframes',
    'scripts.generate_template',
    'scripts.extract_pose',
    'scripts.template_utils',
    'scripts.score_history',
    'scripts.compare_with_lm',
    'mediapipe',
    'cv2',
    'numpy',
    'yaml',
    'websockets',
    'websockets.legacy',
    'websockets.legacy.server',
    'pyttsx3',
    'pyttsx3.drivers',
    'pyttsx3.drivers.sapi5',
    'matplotlib.backends.backend_tkagg',
    'asyncio',
    'PIL',
]

a = Analysis(
    ['launcher.py'],
    pathex=[root],
    binaries=mp_binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'tensorflow', 'webots'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='AI健身教练',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
