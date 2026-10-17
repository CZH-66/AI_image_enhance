# AI Image Enhancer Web Starter

## 文件放置
将 `web_app.py` 放在现有项目根目录，与 `ai_engine.py` 同级；将 `web/index.html` 放入项目根目录的 `web` 文件夹。不要覆盖 `gui.py` 或 `ai_engine.py`。

## 安装依赖
在项目根目录 PowerShell 中运行：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install fastapi uvicorn python-multipart
```

## 启动

```powershell
python -m uvicorn web_app:app --host 127.0.0.1 --port 8000
```

打开 http://127.0.0.1:8000 使用网页，或打开 http://127.0.0.1:8000/docs 测试 API。

## 注意
这是初始适配版，尚未在你的电脑上用实际 `ai_engine.py` 运行验证。它会按参数名适配 `AIEnhancer.enhance()`；如果提示参数不匹配或找不到输出文件，请提供当前 `ai_engine.py`，按真实函数签名微调。上传限制 20 MB，推理请求串行执行以避免同一 GPU 显存冲突。结果保存在 `web_outputs/`。目前仅用于本机测试，不要直接暴露到公网。
