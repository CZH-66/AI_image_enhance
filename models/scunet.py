import os
import cv2
import torch
import numpy as np

from PIL import Image

try:
    from spandrel import ModelLoader, ImageModelDescriptor
except ImportError:
    ModelLoader = None
    ImageModelDescriptor = None


class SCUNetDenoiser:
    """
    SCUNet AI 图像降噪器

    特点：
    1. SCUNet 深度学习降噪
    2. 正确处理 OpenCV BGR / 模型 RGB
    3. 原图融合，减少细节损失
    4. 颜色保护，降低色调漂移
    5. 支持 CUDA / CPU
    6. 支持大图分块处理
    """

    def __init__(
        self,
        model_path,
        device=None,
        tile_size=256,
        tile_overlap=16
    ):
        self.model_path = model_path

        # 自动选择设备
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        self.tile_size = tile_size
        self.tile_overlap = tile_overlap

        self.model = None

    # =========================================================
    # 加载模型
    # =========================================================

    def load(self):
        """
        加载 SCUNet 模型。
        """

        if self.model is not None:
            return

        if ModelLoader is None:
            raise ImportError(
                "未安装 spandrel，请执行：\n"
                "python -m pip install spandrel"
            )

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"SCUNet 模型不存在：\n{self.model_path}"
            )

        print("=" * 60)
        print("正在加载 SCUNet AI 降噪模型...")
        print(f"模型：{self.model_path}")
        print(f"设备：{self.device}")
        print("=" * 60)

        loader = ModelLoader()

        self.model = loader.load_from_file(
            self.model_path
        )

        # 检查是否为图像模型
        if ImageModelDescriptor is not None:
            if not isinstance(
                self.model,
                ImageModelDescriptor
            ):
                raise RuntimeError(
                    "加载的模型不是有效的图像模型"
                )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

        print("SCUNet 模型加载完成")

    # =========================================================
    # BGR numpy → RGB Tensor
    # =========================================================

    def _numpy_to_tensor(self, image_bgr):
        """
        OpenCV BGR uint8
        ↓
        RGB float32 Tensor

        输入：
            H × W × 3
            uint8

        输出：
            1 × 3 × H × W
            float32
            0~1
        """

        if image_bgr.ndim != 3:
            raise ValueError(
                "SCUNet 输入必须是 H×W×3 彩色图像"
            )

        if image_bgr.shape[2] != 3:
            raise ValueError(
                "SCUNet 当前只支持 RGB/BGR 三通道图像"
            )

        # -----------------------------------------------------
        # 这里只进行一次 BGR → RGB
        # -----------------------------------------------------

        image_rgb = image_bgr[:, :, ::-1].copy()

        image_rgb = image_rgb.astype(
            np.float32
        ) / 255.0

        tensor = torch.from_numpy(
            image_rgb.transpose(2, 0, 1)
        ).unsqueeze(0)

        return tensor

    # =========================================================
    # RGB Tensor → BGR numpy
    # =========================================================

    @staticmethod
    def _tensor_to_bgr(tensor):
        """
        SCUNet 输出：
            1 × 3 × H × W
            RGB

        转换为：
            H × W × 3
            BGR uint8
        """

        tensor = tensor.detach().float().cpu()

        tensor = tensor.squeeze(0)

        tensor = torch.clamp(
            tensor,
            0.0,
            1.0
        )

        image_rgb = tensor.numpy()

        image_rgb = image_rgb.transpose(
            1,
            2,
            0
        )

        image_rgb = (
            image_rgb * 255.0 + 0.5
        ).astype(np.uint8)

        # -----------------------------------------------------
        # RGB → BGR
        # -----------------------------------------------------

        image_bgr = image_rgb[
            :, :, ::-1
        ].copy()

        return image_bgr

    # =========================================================
    # 单块推理
    # =========================================================

    def _process_full(self, image_bgr):
        """
        对一张 BGR 图像进行 SCUNet 推理。
        """

        tensor = self._numpy_to_tensor(
            image_bgr
        )

        tensor = tensor.to(
            self.device
        )

        with torch.no_grad():

            output = self.model(
                tensor
            )

        return self._tensor_to_bgr(
            output
        )

    # =========================================================
    # 大图分块推理
    # =========================================================

    def _process_tiled(self, image_bgr):
        """
        大图分块处理。

        使用重叠区域降低拼接痕迹。
        """

        h, w = image_bgr.shape[:2]

        tile = self.tile_size
        overlap = self.tile_overlap

        # 小图直接处理
        if h <= tile and w <= tile:

            return self._process_full(
                image_bgr
            )

        stride = tile - overlap

        # 输出累加
        output = np.zeros(
            (h, w, 3),
            dtype=np.float32
        )

        # 权重
        weight = np.zeros(
            (h, w, 1),
            dtype=np.float32
        )

        y = 0

        while y < h:

            y1 = min(
                y,
                max(0, h - tile)
            )

            y2 = min(
                y1 + tile,
                h
            )

            x = 0

            while x < w:

                x1 = min(
                    x,
                    max(0, w - tile)
                )

                x2 = min(
                    x1 + tile,
                    w
                )

                crop = image_bgr[
                    y1:y2,
                    x1:x2
                ]

                result = self._process_full(
                    crop
                )

                rh, rw = result.shape[:2]

                output[
                    y1:y1 + rh,
                    x1:x1 + rw
                ] += result.astype(
                    np.float32
                )

                weight[
                    y1:y1 + rh,
                    x1:x1 + rw
                ] += 1.0

                if x + stride >= w:
                    break

                x += stride

            if y + stride >= h:
                break

            y += stride

        output /= np.maximum(
            weight,
            1e-6
        )

        output = np.clip(
            output,
            0,
            255
        ).astype(np.uint8)

        return output

    # =========================================================
    # 颜色保护
    # =========================================================

    @staticmethod
    def _protect_color(
        original,
        denoised,
        strength
    ):
        """
        降低 AI 降噪导致的颜色漂移。

        思路：

        1. SCUNet 主要负责去除噪声
        2. 原图提供颜色参考
        3. 对颜色通道进行适度保护

        strength：
            1~10
        """

        # -----------------------------------------------------
        # 转换到 LAB 色彩空间
        # -----------------------------------------------------

        original_lab = cv2.cvtColor(
            original,
            cv2.COLOR_BGR2LAB
        )

        denoised_lab = cv2.cvtColor(
            denoised,
            cv2.COLOR_BGR2LAB
        )

        # -----------------------------------------------------
        # L：亮度
        # A/B：颜色
        #
        # 我们主要采用 SCUNet 的亮度结果，
        # 但对颜色进行原图保护。
        # -----------------------------------------------------

        # 强度越高，允许 AI 对颜色影响稍微增加
        color_ai_ratio = (
            0.08 + strength * 0.012
        )

        color_ai_ratio = max(
            0.08,
            min(
                color_ai_ratio,
                0.20
            )
        )

        color_original_ratio = (
            1.0 - color_ai_ratio
        )

        result_lab = denoised_lab.copy()

        # L 通道使用 AI 结果
        result_lab[:, :, 0] = (
            denoised_lab[:, :, 0]
        )

        # A/B 通道更多保留原图颜色
        result_lab[:, :, 1] = (
            denoised_lab[:, :, 1].astype(
                np.float32
            ) * color_ai_ratio
            +
            original_lab[:, :, 1].astype(
                np.float32
            ) * color_original_ratio
        )

        result_lab[:, :, 2] = (
            denoised_lab[:, :, 2].astype(
                np.float32
            ) * color_ai_ratio
            +
            original_lab[:, :, 2].astype(
                np.float32
            ) * color_original_ratio
        )

        result_lab = np.clip(
            result_lab,
            0,
            255
        ).astype(np.uint8)

        result = cv2.cvtColor(
            result_lab,
            cv2.COLOR_LAB2BGR
        )

        return result

    # =========================================================
    # 原图融合
    # =========================================================

    @staticmethod
    def _blend_original(
        original,
        denoised,
        strength
    ):
        """
        AI 降噪结果和原图进行融合。

        强度越低：
            保留更多原图细节

        强度越高：
            使用更多 AI 降噪结果
        """

        # -----------------------------------------------------
        # 原图保留比例
        #
        # 1 → 约 92%
        # 5 → 约 68%
        # 10 → 约 48%
        # -----------------------------------------------------

        preserve_ratio = (
            0.95 - strength * 0.047
        )

        preserve_ratio = max(
            0.45,
            min(
                preserve_ratio,
                0.92
            )
        )

        ai_ratio = (
            1.0 - preserve_ratio
        )

        result = (
            original.astype(np.float32)
            * preserve_ratio
            +
            denoised.astype(np.float32)
            * ai_ratio
        )

        result = np.clip(
            result,
            0,
            255
        ).astype(np.uint8)

        return result

    # =========================================================
    # AI 降噪
    # =========================================================

    def denoise(
        self,
        image,
        strength=5,
        preserve_original=True
    ):
        """
        SCUNet AI 降噪。

        参数：
            image：
                OpenCV BGR uint8 图像

            strength：
                1~10

            preserve_original：
                是否保留原始细节

        返回：
            OpenCV BGR uint8 图像
        """

        self.load()

        strength = max(
            1,
            min(
                int(strength),
                10
            )
        )

        if image is None:
            raise ValueError(
                "输入图像为空"
            )

        if image.ndim != 3:
            raise ValueError(
                "SCUNet 需要彩色图像"
            )

        if image.shape[2] != 3:
            raise ValueError(
                "SCUNet 当前只支持三通道彩色图像"
            )

        # -----------------------------------------------------
        # 原始 BGR 图像
        # -----------------------------------------------------

        original = image.copy()

        print()
        print("-" * 60)
        print(
            f"SCUNet AI 降噪开始"
        )
        print(
            f"降噪强度：{strength}/10"
        )
        print(
            f"设备：{self.device}"
        )
        print("-" * 60)

        # -----------------------------------------------------
        # SCUNet
        #
        # 注意：
        # 这里直接传入 BGR。
        #
        # _numpy_to_tensor()
        # 内部负责唯一一次 BGR → RGB。
        # -----------------------------------------------------

        ai_result = self._process_tiled(
            original
        )

        # -----------------------------------------------------
        # 颜色保护
        # -----------------------------------------------------

        ai_result = self._protect_color(
            original,
            ai_result,
            strength
        )

        # -----------------------------------------------------
        # 原图融合
        # -----------------------------------------------------

        if preserve_original:

            result = self._blend_original(
                original,
                ai_result,
                strength
            )

        else:

            result = ai_result

        print(
            "SCUNet AI 降噪完成"
        )
        print("-" * 60)

        return result

    # =========================================================
    # 释放模型
    # =========================================================

    def clear(self):

        if self.model is not None:

            del self.model

            self.model = None

        if torch.cuda.is_available():

            torch.cuda.empty_cache()


# =============================================================
# 测试
# =============================================================

if __name__ == "__main__":

    import cv2

    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    MODEL_PATH = os.path.join(
        BASE_DIR,
        "weights",
        "scunet_color_real_psnr.pth"
    )

    INPUT_PATH = os.path.join(
        BASE_DIR,
        "input",
        "test.jpg"
    )

    OUTPUT_PATH = os.path.join(
        BASE_DIR,
        "output",
        "test_scunet_fixed.jpg"
    )

    print()
    print("=" * 60)
    print("SCUNet AI 降噪测试")
    print("=" * 60)

    print(
        f"模型：{MODEL_PATH}"
    )

    print(
        f"输入：{INPUT_PATH}"
    )

    print(
        f"输出：{OUTPUT_PATH}"
    )

    # ---------------------------------------------------------
    # Pillow 读取
    #
    # 解决 Windows 中文路径问题
    # ---------------------------------------------------------

    pil_image = Image.open(
        INPUT_PATH
    ).convert("RGB")

    rgb_image = np.array(
        pil_image
    )

    # RGB → BGR
    image = rgb_image[
        :, :, ::-1
    ].copy()

    # ---------------------------------------------------------
    # 创建 SCUNet
    # ---------------------------------------------------------

    denoiser = SCUNetDenoiser(
        model_path=MODEL_PATH,
        device=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
        tile_size=256,
        tile_overlap=16
    )

    # ---------------------------------------------------------
    # AI 降噪
    # ---------------------------------------------------------

    result = denoiser.denoise(
        image,
        strength=5,
        preserve_original=True
    )

    # ---------------------------------------------------------
    # BGR → RGB
    # ---------------------------------------------------------

    result_rgb = result[
        :, :, ::-1
    ].copy()

    output_image = Image.fromarray(
        result_rgb
    )

    # ---------------------------------------------------------
    # 保存
    # ---------------------------------------------------------

    output_image.save(
        OUTPUT_PATH,
        quality=95
    )

    print()
    print("=" * 60)
    print("测试完成")
    print(
        f"输出文件：{OUTPUT_PATH}"
    )
    print("=" * 60)

