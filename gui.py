# -*- coding: utf-8 -*-

"""
AI 图像增强 GUI
- 单张 / 批量
- 写实 / 动漫
- 2× / 4× / 8× / 16×
- SCUNet 降噪
- 进度条
- 图片预览
- 中文路径
"""

import os
import sys
import subprocess

from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QComboBox,
    QCheckBox, QSlider, QProgressBar, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QSpinBox
)

from ai_engine import AIEnhancer, BASE_DIR


# ============================================================
# 图片预览
# ============================================================

class ImagePreview(QLabel):

    def __init__(self, text="暂无图片"):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(360, 300)
        self.setText(text)
        self.setStyleSheet(
            "QLabel {"
            "border: 1px solid #bdbdbd;"
            "background: #f5f5f5;"
            "color: #666;"
            "}"
        )
        self._pixmap = None

    def set_image(self, path):
        if not path or not os.path.exists(path):
            self._pixmap = None
            self.setText("暂无图片")
            return

        pixmap = QPixmap(path)
        if pixmap.isNull():
            self._pixmap = None
            self.setText("无法预览")
            return

        self._pixmap = pixmap
        self._refresh()

    def _refresh(self):
        if self._pixmap is None:
            return

        self.setPixmap(
            self._pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()


# ============================================================
# 后台线程
# ============================================================

class EnhanceWorker(QThread):

    progress = Signal(int)
    status = Signal(str)
    finished_result = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        enhancer,
        mode,
        input_paths,
        output_dir,
        scale,
        style,
        denoise,
        denoise_strength
    ):
        super().__init__()

        self.enhancer = enhancer
        self.mode = mode
        self.input_paths = input_paths
        self.output_dir = output_dir
        self.scale = scale
        self.style = style
        self.denoise = denoise
        self.denoise_strength = denoise_strength

    def single_callback(self, current, total, message):
        percent = int(current / max(1, total) * 100)
        self.progress.emit(percent)
        self.status.emit(str(message))

    def batch_callback(
        self,
        current,
        total,
        filename,
        message,
        total_progress
    ):
        self.progress.emit(int(total_progress))
        self.status.emit(
            f"[{current}/{total}] {message}"
        )

    def run(self):
        try:
            if self.mode == "single":
                result = self.enhancer.enhance(
                    input_path=self.input_paths[0],
                    output_path=None,
                    scale=self.scale,
                    style=self.style,
                    denoise=self.denoise,
                    denoise_strength=self.denoise_strength,
                    progress_callback=self.single_callback
                )
            else:
                result = self.enhancer.enhance_batch(
                    input_paths=self.input_paths,
                    output_dir=self.output_dir,
                    scale=self.scale,
                    style=self.style,
                    denoise=self.denoise,
                    denoise_strength=self.denoise_strength,
                    progress_callback=self.batch_callback
                )

            self.finished_result.emit(result)

        except Exception as e:
            self.error.emit(str(e))


# ============================================================
# 主窗口
# ============================================================

class MainWindow(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("AI 图像增强系统")
        self.resize(1050, 720)

        self.enhancer = AIEnhancer()
        self.worker = None

        self.mode = "single"
        self.input_paths = []

        self.output_dir = os.path.join(
            BASE_DIR, "output"
        )
        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

        self._build_ui()

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------

    def _build_ui(self):

        root = QVBoxLayout(self)

        title = QLabel("AI 图像增强系统")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "font-size: 24px; font-weight: bold; padding: 12px;"
        )
        root.addWidget(title)

        # ====================================================
        # 文件操作
        # ====================================================

        file_box = QGroupBox("输入与输出")
        file_layout = QGridLayout()

        self.input_label = QLabel("未选择图片")
        self.input_label.setWordWrap(True)

        self.select_single_btn = QPushButton("选择图片")
        self.select_single_btn.clicked.connect(
            self.select_single
        )

        self.select_batch_btn = QPushButton("选择多张图片")
        self.select_batch_btn.clicked.connect(
            self.select_batch
        )

        self.output_label = QLabel(
            f"输出目录：{self.output_dir}"
        )
        self.output_label.setWordWrap(True)

        self.output_btn = QPushButton("选择输出目录")
        self.output_btn.clicked.connect(
            self.select_output_dir
        )

        self.open_output_btn = QPushButton("打开输出目录")
        self.open_output_btn.clicked.connect(
            self.open_output_dir
        )

        file_layout.addWidget(
            QLabel("输入："), 0, 0
        )
        file_layout.addWidget(
            self.input_label, 0, 1, 1, 3
        )
        file_layout.addWidget(
            self.select_single_btn, 0, 4
        )
        file_layout.addWidget(
            self.select_batch_btn, 0, 5
        )

        file_layout.addWidget(
            self.output_label, 1, 1, 1, 3
        )
        file_layout.addWidget(
            self.output_btn, 1, 4
        )
        file_layout.addWidget(
            self.open_output_btn, 1, 5
        )

        file_box.setLayout(file_layout)
        root.addWidget(file_box)

        # ====================================================
        # 参数
        # ====================================================

        option_box = QGroupBox("增强参数")
        option_layout = QGridLayout()

        self.style_combo = QComboBox()
        self.style_combo.addItem("写实", "realistic")
        self.style_combo.addItem("动漫", "anime")

        self.scale_combo = QComboBox()
        self.scale_combo.addItem("2×", 2)
        self.scale_combo.addItem("4×", 4)
        self.scale_combo.addItem("8×", 8)
        self.scale_combo.addItem("16×", 16)
        self.scale_combo.setCurrentIndex(1)

        self.denoise_check = QCheckBox("启用 AI 降噪")
        self.denoise_check.stateChanged.connect(
            self.denoise_changed
        )

        self.denoise_slider = QSlider(Qt.Horizontal)
        self.denoise_slider.setMinimum(1)
        self.denoise_slider.setMaximum(10)
        self.denoise_slider.setValue(5)
        self.denoise_slider.setEnabled(False)

        self.denoise_value = QLabel("5")

        self.denoise_slider.valueChanged.connect(
            lambda v: self.denoise_value.setText(str(v))
        )

        option_layout.addWidget(
            QLabel("图像风格："), 0, 0
        )
        option_layout.addWidget(
            self.style_combo, 0, 1
        )

        option_layout.addWidget(
            QLabel("放大倍数："), 0, 2
        )
        option_layout.addWidget(
            self.scale_combo, 0, 3
        )

        option_layout.addWidget(
            self.denoise_check, 1, 0
        )
        option_layout.addWidget(
            self.denoise_slider, 1, 1, 1, 2
        )
        option_layout.addWidget(
            self.denoise_value, 1, 3
        )

        option_box.setLayout(option_layout)
        root.addWidget(option_box)

        # ====================================================
        # 预览
        # ====================================================

        preview_layout = QHBoxLayout()

        left_box = QVBoxLayout()
        left_box.addWidget(
            QLabel("原图")
        )

        self.input_preview = ImagePreview()
        left_box.addWidget(
            self.input_preview
        )

        right_box = QVBoxLayout()
        right_box.addWidget(
            QLabel("增强结果")
        )

        self.output_preview = ImagePreview()
        right_box.addWidget(
            self.output_preview
        )

        preview_layout.addLayout(left_box)
        preview_layout.addLayout(right_box)

        root.addLayout(preview_layout)

        # ====================================================
        # 操作按钮
        # ====================================================

        button_layout = QHBoxLayout()

        self.start_btn = QPushButton("开始处理")
        self.start_btn.setMinimumHeight(42)
        self.start_btn.clicked.connect(
            self.start_process
        )

        self.clear_btn = QPushButton("清空")
        self.clear_btn.setMinimumHeight(42)
        self.clear_btn.clicked.connect(
            self.clear_all
        )

        button_layout.addWidget(
            self.start_btn
        )
        button_layout.addWidget(
            self.clear_btn
        )

        root.addLayout(button_layout)

        # ====================================================
        # 进度
        # ====================================================

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        root.addWidget(
            self.progress_bar
        )

        self.status_label = QLabel(
            "就绪"
        )
        root.addWidget(
            self.status_label
        )

        self.info_label = QLabel(
            self._device_text()
        )
        root.addWidget(
            self.info_label
        )

    # --------------------------------------------------------
    # 设备信息
    # --------------------------------------------------------

    def _device_text(self):
        info = self.enhancer.get_device_info()

        if info.get("device") == "CUDA":
            return (
                f"GPU：{info.get('name', 'CUDA')}    "
                f"显存：{info.get('total_memory', 0):.1f} GB"
            )

        return "当前设备：CPU"

    # --------------------------------------------------------
    # 降噪
    # --------------------------------------------------------

    def denoise_changed(self, state):
        enabled = bool(state)
        self.denoise_slider.setEnabled(
            enabled
        )

    # --------------------------------------------------------
    # 选择单张
    # --------------------------------------------------------

    def select_single(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "图片 (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)"
        )

        if not path:
            return

        self.mode = "single"
        self.input_paths = [path]

        self.input_label.setText(path)
        self.input_preview.set_image(path)

        self.status_label.setText(
            "已选择单张图片"
        )

    # --------------------------------------------------------
    # 选择多张
    # --------------------------------------------------------

    def select_batch(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择多张图片",
            "",
            "图片 (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff)"
        )

        if not paths:
            return

        self.mode = "batch"
        self.input_paths = paths

        self.input_label.setText(
            f"已选择 {len(paths)} 张图片\n"
            + "\n".join(paths[:5])
            + ("..." if len(paths) > 5 else "")
        )

        self.input_preview.set_image(
            paths[0]
        )

        self.status_label.setText(
            f"已选择 {len(paths)} 张图片"
        )

    # --------------------------------------------------------
    # 输出目录
    # --------------------------------------------------------

    def select_output_dir(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "选择输出目录",
            self.output_dir
        )

        if path:
            self.output_dir = path
            self.output_label.setText(
                f"输出目录：{path}"
            )

    def open_output_dir(self):
        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

        try:
            if sys.platform.startswith("win"):
                os.startfile(self.output_dir)
            elif sys.platform == "darwin":
                subprocess.Popen(
                    ["open", self.output_dir]
                )
            else:
                subprocess.Popen(
                    ["xdg-open", self.output_dir]
                )
        except Exception as e:
            QMessageBox.warning(
                self,
                "打开失败",
                str(e)
            )

    # --------------------------------------------------------
    # 开始处理
    # --------------------------------------------------------

    def start_process(self):

        if not self.input_paths:
            QMessageBox.warning(
                self,
                "提示",
                "请先选择图片。"
            )
            return

        scale = self.scale_combo.currentData()
        style = self.style_combo.currentData()
        denoise = self.denoise_check.isChecked()
        strength = self.denoise_slider.value()

        # 16× 提醒：仍然允许直接处理
        if scale == 16:
            answer = QMessageBox.question(
                self,
                "16× 超分辨率",
                "16×采用两阶段4× AI 超分辨率，"
                "处理时间和显存占用会明显增加。\n\n"
                "是否继续？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )

            if answer != QMessageBox.Yes:
                return

        self.progress_bar.setValue(0)
        self.start_btn.setEnabled(False)
        self.select_single_btn.setEnabled(False)
        self.select_batch_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)

        self.status_label.setText(
            "正在启动 AI 模型..."
        )

        self.worker = EnhanceWorker(
            enhancer=self.enhancer,
            mode=self.mode,
            input_paths=self.input_paths,
            output_dir=self.output_dir,
            scale=scale,
            style=style,
            denoise=denoise,
            denoise_strength=strength
        )

        self.worker.progress.connect(
            self.progress_bar.setValue
        )
        self.worker.status.connect(
            self.status_label.setText
        )
        self.worker.finished_result.connect(
            self.process_finished
        )
        self.worker.error.connect(
            self.process_error
        )
        self.worker.start()

    # --------------------------------------------------------
    # 完成
    # --------------------------------------------------------

    def process_finished(self, result):

        self.start_btn.setEnabled(True)
        self.select_single_btn.setEnabled(True)
        self.select_batch_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)

        if self.mode == "single":

            if isinstance(result, dict) and result.get("success"):

                output_path = result.get(
                    "output_path"
                )

                self.progress_bar.setValue(100)
                self.status_label.setText(
                    "处理完成"
                )

                self.output_preview.set_image(
                    output_path
                )

                QMessageBox.information(
                    self,
                    "处理完成",
                    "图片增强完成！\n\n"
                    f"输出文件：\n{output_path}"
                )

            else:

                error = (
                    result.get("error", "未知错误")
                    if isinstance(result, dict)
                    else str(result)
                )

                self.process_error(error)

        else:

            success_count = sum(
                1 for r in result
                if r.get("success")
            )

            fail_count = len(result) - success_count

            self.progress_bar.setValue(100)
            self.status_label.setText(
                f"批量处理完成：成功 {success_count} 张，失败 {fail_count} 张"
            )

            if result:
                first_success = next(
                    (
                        r for r in result
                        if r.get("success")
                    ),
                    None
                )

                if first_success:
                    self.output_preview.set_image(
                        first_success.get("output_path")
                    )

            QMessageBox.information(
                self,
                "批量处理完成",
                f"成功：{success_count} 张\n"
                f"失败：{fail_count} 张\n\n"
                f"输出目录：\n{self.output_dir}"
            )

        self.worker = None

    # --------------------------------------------------------
    # 错误
    # --------------------------------------------------------

    def process_error(self, message):

        self.start_btn.setEnabled(True)
        self.select_single_btn.setEnabled(True)
        self.select_batch_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)

        self.status_label.setText(
            "处理失败"
        )

        QMessageBox.critical(
            self,
            "处理失败",
            f"图像处理失败：\n\n{message}"
        )

        self.worker = None

    # --------------------------------------------------------
    # 清空
    # --------------------------------------------------------

    def clear_all(self):

        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(
                self,
                "提示",
                "当前正在处理，请等待处理结束。"
            )
            return

        self.input_paths = []
        self.mode = "single"

        self.input_label.setText(
            "未选择图片"
        )

        self.input_preview.set_image(None)
        self.output_preview.set_image(None)

        self.progress_bar.setValue(0)
        self.status_label.setText(
            "就绪"
        )

    # --------------------------------------------------------
    # 关闭
    # --------------------------------------------------------

    def closeEvent(self, event):

        if self.worker is not None and self.worker.isRunning():

            QMessageBox.warning(
                self,
                "提示",
                "当前正在处理，请等待处理完成后再关闭程序。"
            )

            event.ignore()
            return

        try:
            self.enhancer.clear()
        except Exception:
            pass

        event.accept()


# ============================================================
# 启动
# ============================================================

def main():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(
        app.exec()
    )


if __name__ == "__main__":
    main()
