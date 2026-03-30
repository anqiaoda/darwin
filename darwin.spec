# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[
        ('F:\\miniforge3\\Lib\\site-packages\\glfw\\glfw3.dll', '.'),
        ('F:\\miniforge3\\Lib\\site-packages\\mujoco\\mujoco.dll', '.'),
        ('F:\\miniforge3\\Lib\\site-packages\\mujoco\\plugin\\', 'mujoco/plugin'),
    ],
    datas=[
        ('config.json', '.'),
        ('data', 'data'),
        ('./unitree_robots', 'unitree_robots'),
    ],
    hiddenimports=[
        'cv2',
        'numpy',
        'requests',
        'mujoco',
        'mujoco._functions',
        'mujoco._render',
        'mujoco._render.ffi',
        'glfw',
        'mujoco.mjx',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name='darwin-v1.2',
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