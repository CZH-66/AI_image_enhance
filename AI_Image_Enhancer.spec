# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_data_files


# ============================================================
# 1. 项目路径
# ============================================================

BASE_DIR = os.path.abspath(".")

REALESRGAN_DIR = os.path.join(
    BASE_DIR,
    "Real-ESRGAN-master",
    "Real-ESRGAN-master"
)

REALESRGAN_PACKAGE = os.path.join(
    REALESRGAN_DIR,
    "realesrgan"
)

BASICSR_DIR = os.path.join(
    BASE_DIR,
    ".venv",
    "Lib",
    "site-packages",
    "basicsr"
)

WEIGHTS_DIR = os.path.join(
    BASE_DIR,
    "weights"
)


# ============================================================
# 2. 路径检查
# ============================================================

print("=" * 70)
print("AI Image Enhancer - PyInstaller")
print("=" * 70)

print("BASE_DIR      =", BASE_DIR)
print("REALESRGAN_DIR=", REALESRGAN_DIR)
print("BASICSR_DIR   =", BASICSR_DIR)
print("WEIGHTS_DIR   =", WEIGHTS_DIR)

print("=" * 70)


if not os.path.exists(REALESRGAN_PACKAGE):
    raise FileNotFoundError(
        "找不到 Real-ESRGAN：\n" +
        REALESRGAN_PACKAGE
    )

if not os.path.exists(BASICSR_DIR):
    raise FileNotFoundError(
        "找不到 BasicSR：\n" +
        BASICSR_DIR
    )

if not os.path.exists(WEIGHTS_DIR):
    raise FileNotFoundError(
        "找不到 weights：\n" +
        WEIGHTS_DIR
    )


# ============================================================
# 3. 数据文件
# ============================================================

datas = []


# ============================================================
# 3.1 模型权重
# ============================================================

WEIGHT_FILES = [
    "RealESRGAN_x4plus.pth",
    "RealESRGAN_x4plus_anime_6B.pth",
    "scunet_color_real_psnr.pth",
]


for weight_name in WEIGHT_FILES:

    weight_path = os.path.join(
        WEIGHTS_DIR,
        weight_name
    )

    if not os.path.exists(weight_path):

        raise FileNotFoundError(
            "找不到模型：\n" +
            weight_path
        )

    print("添加模型：", weight_path)

    datas.append(
        (
            weight_path,
            "weights"
        )
    )


# ============================================================
# 3.2 Real-ESRGAN
# ============================================================

datas.append(
    (
        REALESRGAN_PACKAGE,
        "realesrgan"
    )
)


# ============================================================
# 3.3 BasicSR
# ============================================================

datas.append(
    (
        BASICSR_DIR,
        "basicsr"
    )
)


# ============================================================
# 3.4 Spandrel
# ============================================================

try:

    datas += collect_data_files(
        "spandrel"
    )

except Exception:

    pass


# ============================================================
# 4. Hidden Imports
# ============================================================

hiddenimports = [

    # PySide6
    "PySide6",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",

    # PyTorch
    "torch",
    "torch._C",
    "torch.cuda",
    "torch.version",

    # torchvision
    "torchvision",
    "torchvision.transforms",
    "torchvision.transforms.functional",

    # BasicSR
    "basicsr",
    "basicsr.archs",
    "basicsr.archs.rrdbnet_arch",
    "basicsr.data",
    "basicsr.models",
    "basicsr.ops",
    "basicsr.ops.fused_act",
    "basicsr.utils",

    # Real-ESRGAN
    "realesrgan",
    "realesrgan.archs",
    "realesrgan.archs.srvgg_arch",
    "realesrgan.utils",

    # Spandrel
    "spandrel",
]


# ============================================================
# 5. Analysis
# ============================================================

a = Analysis(

    [
        os.path.join(
            BASE_DIR,
            "gui.py"
        )
    ],

    pathex=[
        BASE_DIR,
        REALESRGAN_DIR,
    ],

    binaries=[],

    datas=datas,

    hiddenimports=hiddenimports,

    hookspath=[],

    hooksconfig={},

    runtime_hooks=[],

    excludes=[

        "IPython",
        "ipykernel",
        "jupyter",
        "jupyterlab",
        "notebook",

        "pandas",

        "matplotlib",

        "tensorflow",
        "tensorflow_core",
        "keras",

        "pygame",
        "onnxruntime",
    ],

    noarchive=False,
)


# ============================================================
# 6. PYZ
# ============================================================

pyz = PYZ(
    a.pure
)


# ============================================================
# 7. EXE
# ============================================================

exe = EXE(

    pyz,

    a.scripts,

    [],

    exclude_binaries=True,

    name="AI_Image_Enhancer",

    debug=False,

    bootloader_ignore_signals=False,

    strip=False,

    upx=False,

    console=False,

    disable_windowed_traceback=False,

    argv_emulation=False,

    target_arch=None,

    codesign_identity=None,

    entitlements_file=None,

    contents_directory=".",

)


# ============================================================
# 8. COLLECT
# ============================================================

coll = COLLECT(

    exe,

    a.binaries,

    a.datas,

    strip=False,

    upx=False,

    name="AI_Image_Enhancer",

)