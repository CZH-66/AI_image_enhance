# -*- coding: utf-8 -*-
"""
AI 图像增强引擎
功能：
- Real-ESRGAN：写实 / 动漫
- 2× / 4× / 8× / 16×
- 16×采用两次4× Real-ESRGAN 级联
- SCUNet AI 降噪
- CUDA 加速
- RTX 4060 8GB 自动 tile 降级
- 中文路径
- 单张 / 批量
- 兼容 progress_callback / callback
"""

import os
import sys
import gc
import traceback

# 尽量降低 Windows CUDA 内存碎片
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch

# ============================================================
# 路径
# ============================================================

def get_base_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = get_base_dir()

REALESRGAN_DIR = os.path.join(
    BASE_DIR, "Real-ESRGAN-master", "Real-ESRGAN-master"
)

WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

REAL_MODEL_PATH = os.path.join(
    WEIGHTS_DIR, "RealESRGAN_x4plus.pth"
)
ANIME_MODEL_PATH = os.path.join(
    WEIGHTS_DIR, "RealESRGAN_x4plus_anime_6B.pth"
)
SCUNET_MODEL_PATH = os.path.join(
    WEIGHTS_DIR, "scunet_color_real_psnr.pth"
)

if REALESRGAN_DIR not in sys.path:
    sys.path.insert(0, REALESRGAN_DIR)

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from models.scunet import SCUNetDenoiser

SUPPORTED_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
)

# ============================================================
# AIEnhancer
# ============================================================

class AIEnhancer:

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.half = False  # 当前环境 FP16 已测试失败，保持 False
        self.upsampler = None
        self.current_style = None
        self.current_tile = 256
        self.scunet = None
        self._cancel_flag = False

        print("=" * 70)
        print("AI 图像增强引擎")
        print("=" * 70)
        print("项目目录：", BASE_DIR)
        print("模型目录：", WEIGHTS_DIR)
        print("写实模型：", REAL_MODEL_PATH)
        print("动漫模型：", ANIME_MODEL_PATH)
        print("SCUNet：", SCUNET_MODEL_PATH)
        print("设备：", self.device)

        if torch.cuda.is_available():
            try:
                print("GPU：", torch.cuda.get_device_name(0))
                print(
                    "显存：%.2f GB" %
                    (torch.cuda.get_device_properties(0).total_memory / 1024**3)
                )
            except Exception:
                pass
        print("=" * 70)

    def get_device_info(self):
        """返回 GUI 可直接使用的设备信息。"""
        if torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(0)
                return {
                    "device": "CUDA",
                    "name": torch.cuda.get_device_name(0),
                    "total_memory": props.total_memory / 1024**3,
                }
            except Exception:
                return {
                    "device": "CUDA",
                    "name": "NVIDIA GPU",
                    "total_memory": 0,
                }

        return {
            "device": "CPU",
            "name": "CPU",
            "total_memory": 0,
        }

    # --------------------------------------------------------
    # 基础工具
    # --------------------------------------------------------

    @staticmethod
    def is_supported_image(path):
        return bool(path) and str(path).lower().endswith(SUPPORTED_EXTENSIONS)

    @staticmethod
    def _read_image(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"输入图片不存在：\n{path}")

        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            if im.mode == "RGBA":
                arr = np.array(im)
                return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
            arr = np.array(im.convert("RGB"))
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _save_image(image, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        ext = os.path.splitext(path)[1].lower()
        if not ext:
            ext = ".png"
            path += ext

        if ext not in SUPPORTED_EXTENSIONS:
            ext = ".png"

        if ext in (".jpg", ".jpeg") and image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        ok, encoded = cv2.imencode(ext, image)
        if not ok:
            raise RuntimeError(f"图片编码失败：\n{path}")

        with open(path, "wb") as f:
            f.write(encoded.tobytes())

        return path

    @staticmethod
    def generate_output_path(input_path, output_dir, scale, style, denoise):
        os.makedirs(output_dir, exist_ok=True)
        name, ext = os.path.splitext(os.path.basename(input_path))
        style_name = "anime" if str(style).lower() == "anime" else "real"
        dn = "_denoise" if denoise else ""
        return os.path.join(
            output_dir,
            f"{name}_{style_name}_{scale}x{dn}{ext}"
        )

    @staticmethod
    def _clear_cuda_memory():
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass

    @staticmethod
    def _call_progress(callback, current, total, message):
        if callback is None:
            return
        try:
            callback(current, total, message)
        except TypeError:
            try:
                callback(current)
            except Exception:
                pass
        except Exception:
            pass

    @staticmethod
    def _call_batch_progress(
        callback, current, total, filename, message, total_progress
    ):
        if callback is None:
            return
        try:
            callback(
                current, total, filename, message, total_progress
            )
        except TypeError:
            try:
                callback(current, total, message)
            except TypeError:
                try:
                    callback(total_progress)
                except Exception:
                    pass
            except Exception:
                pass
        except Exception:
            pass

    # --------------------------------------------------------
    # Real-ESRGAN
    # --------------------------------------------------------

    def _create_upsampler(self, style="realistic", tile_size=256):
        style = str(style).lower()

        if style == "anime":
            model_path = ANIME_MODEL_PATH
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3,
                num_feat=64, num_block=6,
                num_grow_ch=32, scale=4
            )
        else:
            style = "realistic"
            model_path = REAL_MODEL_PATH
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3,
                num_feat=64, num_block=23,
                num_grow_ch=32, scale=4
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Real-ESRGAN 模型不存在：\n{model_path}\n\n"
                f"请检查项目 weights 文件夹。"
            )

        print(f"[Real-ESRGAN] style={style}, tile={tile_size}")
        print(f"[Real-ESRGAN] model={model_path}")

        self.current_tile = tile_size

        return RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=tile_size,
            tile_pad=10,
            pre_pad=0,
            half=False
        )

    def _release_upsampler(self):
        if self.upsampler is not None:
            try:
                del self.upsampler
            except Exception:
                pass
            self.upsampler = None

        self._clear_cuda_memory()

    def _ensure_upsampler(self, style):
        style = "anime" if str(style).lower() == "anime" else "realistic"

        if self.upsampler is None or self.current_style != style:
            self._release_upsampler()
            self.current_style = style
            self.upsampler = self._create_upsampler(style, 256)

        return self.upsampler

    # --------------------------------------------------------
    # SCUNet
    # --------------------------------------------------------

    def _ensure_scunet(self):
        if self.scunet is None:
            if not os.path.exists(SCUNET_MODEL_PATH):
                raise FileNotFoundError(
                    f"SCUNet 模型不存在：\n{SCUNET_MODEL_PATH}"
                )

            self.scunet = SCUNetDenoiser(
                model_path=SCUNET_MODEL_PATH,
                device=self.device,
                tile_size=256,
                tile_overlap=16
            )
        return self.scunet

    def denoise_image(self, image, strength=5, style="realistic"):
        strength = max(1, min(10, int(strength)))
        if str(style).lower() == "anime":
            effective = max(1, round(strength * 0.75))
        else:
            effective = strength

        print(
            f"[SCUNet] strength={strength}, effective={effective}"
        )

        result = self._ensure_scunet().denoise(
            image,
            strength=effective,
            preserve_original=True
        )

        result = np.asarray(result)
        if result.dtype != np.uint8:
            result = np.clip(result, 0, 255).astype(np.uint8)
        return result

    # --------------------------------------------------------
    # 单次 Real-ESRGAN 推理
    # --------------------------------------------------------

    def _enhance_once(self, image, scale, style, callback, progress_start=55, progress_end=88):
        """
        单次 4×/8× Real-ESRGAN。
        OOM 时自动 256 -> 128 -> 64。
        """
        last_error = None

        for tile in (256, 128, 64):
            self._check_cancel()
            try:
                self._release_upsampler()
                self.upsampler = self._create_upsampler(style, tile)

                self._call_progress(
                    callback, progress_start, 100,
                    f"AI 超分辨率处理中（tile={tile}）..."
                )

                with torch.inference_mode():
                    output, _ = self.upsampler.enhance(
                        image, outscale=scale
                    )

                if output is None:
                    raise RuntimeError("Real-ESRGAN 输出为空。")

                self._call_progress(
                    callback, progress_end, 100,
                    "AI 超分辨率处理完成"
                )
                print(f"[Real-ESRGAN] tile={tile} 成功")
                return output

            except torch.cuda.OutOfMemoryError as e:
                last_error = e
                print(f"[Real-ESRGAN] tile={tile} 显存不足，继续降低 tile")
                self._release_upsampler()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    last_error = e
                    print(f"[Real-ESRGAN] tile={tile} 显存不足")
                    self._release_upsampler()
                else:
                    raise

        raise RuntimeError(
            "GPU 显存不足：tile=256、128、64 均无法完成本阶段超分辨率。\n"
            "请降低输入图片分辨率或关闭其他 GPU 程序。"
        ) from last_error

    # --------------------------------------------------------
    # 16× 级联
    # --------------------------------------------------------

    def _enhance_16x(self, image, style, callback):
        """
        16× = 第一次 4× + 第二次 4×。
        两次推理之间主动释放显存。
        """
        self._call_progress(
            callback, 50, 100,
            "16×增强：第一阶段 4× AI 超分辨率..."
        )

        stage1 = self._enhance_once(
            image=image,
            scale=4,
            style=style,
            callback=callback,
            progress_start=52,
            progress_end=62
        )

        self._call_progress(
            callback, 64, 100,
            "16×增强：正在释放第一阶段显存..."
        )

        # 不再保留第一阶段的 GPU 模型
        self._release_upsampler()
        self._clear_cuda_memory()

        # 中间结果转为连续 uint8 数组
        stage1 = np.ascontiguousarray(
            np.clip(stage1, 0, 255).astype(np.uint8)
        )

        self._call_progress(
            callback, 68, 100,
            "16×增强：第二阶段 4× AI 超分辨率..."
        )

        stage2 = self._enhance_once(
            image=stage1,
            scale=4,
            style=style,
            callback=callback,
            progress_start=70,
            progress_end=90
        )

        del stage1
        self._clear_cuda_memory()

        return stage2

    # --------------------------------------------------------
    # 取消支持
    # --------------------------------------------------------

    def request_cancel(self):
        """由外部（Web/GUI）调用，请求取消当前任务。"""
        self._cancel_flag = True
        print("[AIEnhancer] 收到取消请求")

    def _check_cancel(self):
        if self._cancel_flag:
            raise RuntimeError("任务已被用户取消")

    # --------------------------------------------------------
    # 主接口
    # --------------------------------------------------------

    def enhance(
        self,
        input_path,
        output_path=None,
        scale=4,
        style="realistic",
        denoise=False,
        denoise_strength=5,
        callback=None,
        progress_callback=None,
        **kwargs
    ):
        if progress_callback is not None:
            callback = progress_callback

        if callback is None and "progress" in kwargs:
            callback = kwargs["progress"]

        self._cancel_flag = False

        try:
            self._check_cancel()

            if not input_path:
                raise ValueError("没有选择输入图片。")

            if not os.path.exists(input_path):
                raise FileNotFoundError(
                    f"输入图片不存在：\n{input_path}"
                )

            if not self.is_supported_image(input_path):
                raise ValueError(
                    "不支持的图片格式。\n"
                    "支持 JPG、JPEG、PNG、BMP、WEBP、TIFF。"
                )

            scale = int(scale)
            if scale not in (2, 4, 8, 16):
                raise ValueError("放大倍数只能是 2、4、8 或 16。")

            style = "anime" if str(style).lower() == "anime" else "realistic"

            self._call_progress(
                callback, 5, 100, "正在读取图片..."
            )

            image = self._read_image(input_path)

            alpha = None
            if image.ndim == 3 and image.shape[2] == 4:
                alpha = image[:, :, 3].copy()
                image = image[:, :, :3]

            if denoise:
                self._call_progress(
                    callback, 15, 100, "正在进行 AI 降噪..."
                )
                image = self.denoise_image(
                    image,
                    strength=denoise_strength,
                    style=style
                )
                self._call_progress(
                    callback, 45, 100,
                    "AI 降噪完成，准备超分辨率..."
                )
            else:
                self._call_progress(
                    callback, 30, 100,
                    "准备进行 AI 超分辨率..."
                )

            # 2× / 4× / 8× 保持原有逻辑
            if scale == 16:
                output = self._enhance_16x(
                    image, style, callback
                )
            else:
                output = self._enhance_once(
                    image=image,
                    scale=scale,
                    style=style,
                    callback=callback,
                    progress_start=52,
                    progress_end=88
                )

            if alpha is not None:
                h, w = output.shape[:2]
                alpha2 = cv2.resize(
                    alpha, (w, h),
                    interpolation=cv2.INTER_LINEAR
                )
                output = np.dstack((output, alpha2))

            output = np.asarray(output)
            if output.dtype != np.uint8:
                output = np.clip(output, 0, 255).astype(np.uint8)

            if output_path is None:
                output_path = self.generate_output_path(
                    input_path,
                    os.path.join(BASE_DIR, "output"),
                    scale, style, denoise
                )

            self._call_progress(
                callback, 94, 100, "正在保存增强结果..."
            )
            saved_path = self._save_image(
                output, output_path
            )

            self._call_progress(
                callback, 100, 100, "处理完成"
            )

            return {
                "success": True,
                "input_path": input_path,
                "output_path": saved_path,
                "scale": scale,
                "style": style,
                "denoise": bool(denoise),
                "denoise_strength": int(denoise_strength) if denoise else 0,
                "device": str(self.device),
                "input_size": (image.shape[1], image.shape[0]),
                "output_size": (output.shape[1], output.shape[0]),
                "tile": self.current_tile
            }

        except Exception as e:
            print("\n[AIEnhancer] 处理失败：")
            traceback.print_exc()
            self._clear_cuda_memory()

            return {
                "success": False,
                "input_path": input_path,
                "output_path": output_path,
                "scale": scale,
                "style": style,
                "denoise": bool(denoise),
                "denoise_strength": int(denoise_strength) if denoise else 0,
                "device": str(self.device),
                "error": str(e)
            }

    # --------------------------------------------------------
    # 批量处理
    # --------------------------------------------------------

    def enhance_batch(
        self,
        input_paths,
        output_dir,
        scale=4,
        style="realistic",
        denoise=False,
        denoise_strength=5,
        callback=None,
        progress_callback=None,
        **kwargs
    ):
        if progress_callback is not None:
            callback = progress_callback

        if callback is None and "progress" in kwargs:
            callback = kwargs["progress"]

        valid = [
            p for p in input_paths
            if os.path.isfile(p) and self.is_supported_image(p)
        ]

        total = len(valid)
        results = []

        if total == 0:
            return results

        os.makedirs(output_dir, exist_ok=True)

        for index, input_path in enumerate(valid):
            current = index + 1
            filename = os.path.basename(input_path)

            def single_progress(cur, all_count, message):
                ratio = cur / max(1, all_count)
                total_progress = int(
                    ((index + ratio) / total) * 100
                )
                self._call_batch_progress(
                    callback,
                    current,
                    total,
                    filename,
                    message,
                    total_progress
                )

            output_path = self.generate_output_path(
                input_path,
                output_dir,
                scale,
                style,
                denoise
            )

            result = self.enhance(
                input_path=input_path,
                output_path=output_path,
                scale=scale,
                style=style,
                denoise=denoise,
                denoise_strength=denoise_strength,
                callback=single_progress
            )

            results.append(result)

            self._call_batch_progress(
                callback,
                current,
                total,
                filename,
                "处理完成" if result.get("success") else "处理失败",
                int(current / total * 100)
            )

            self._clear_cuda_memory()

        return results

    def clear(self):
        self._release_upsampler()

        if self.scunet is not None:
            try:
                del self.scunet
            except Exception:
                pass
            self.scunet = None

        self.current_style = None
        self._clear_cuda_memory()


if __name__ == "__main__":
    print("ai_engine.py 已加载。")
    print("支持倍率：2× / 4× / 8× / 16×")
    print("16×：两阶段 4× Real-ESRGAN 级联。")
