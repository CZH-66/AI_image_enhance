from pathlib import Path
from threading import Lock, Thread
import inspect, uuid, time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from io import BytesIO

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "web_outputs"
for folder in (UPLOAD_DIR, OUTPUT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AI 图片超分辨率服务", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
_engine = None
_engine_lock = Lock()
_inference_lock = Lock()
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

_tasks: dict = {}
_tasks_lock = Lock()
MAX_BATCH_FILES = 20


def _validate_params(style: str, scale: int, denoise_strength: int):
    if style not in ("realistic", "anime"):
        raise HTTPException(400, "style 只能是 realistic 或 anime")
    if scale not in (2, 4, 8, 16):
        raise HTTPException(400, "scale 只能是 2、4、8 或 16")
    if not 1 <= denoise_strength <= 10:
        raise HTTPException(400, "denoise_strength 必须在 1 到 10 之间")


def _save_upload(file: UploadFile) -> tuple[str, Path]:
    """读取上传文件，校验格式与大小，转 PNG 存入 uploads，返回 (token, input_path)。"""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        raise HTTPException(400, f"仅支持 PNG、JPG、JPEG、WEBP、BMP：{file.filename}")
    token = uuid.uuid4().hex
    input_path = UPLOAD_DIR / f"{token}.png"
    try:
        content = file.file.read()
        if not content:
            raise HTTPException(400, f"上传文件为空：{file.filename}")
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"图片不能超过 20 MB：{file.filename}")
        try:
            with Image.open(BytesIO(content)) as im:
                im.verify()
            with Image.open(BytesIO(content)) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.save(input_path, format="PNG")
        except (UnidentifiedImageError, OSError, ValueError):
            raise HTTPException(400, f"无法识别该图片：{file.filename}")
    finally:
        file.file.close()
    return token, input_path


def _create_engine():
    from ai_engine import AIEnhancer
    return AIEnhancer()


def get_engine():
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = _create_engine()
    return _engine


def _init_engine_background():
    global _engine
    if _engine is not None:
        return
    engine = _create_engine()
    with _engine_lock:
        if _engine is None:
            _engine = engine


def run_enhance(engine, input_path: Path, output_path: Path, scale: int,
                style: str, denoise: bool, denoise_strength: int) -> Path:
    """根据 enhance() 的参数名适配现有引擎，避免猜测位置参数顺序。"""
    method = engine.enhance
    params = inspect.signature(method).parameters
    aliases = {
        "input_path": ("input_path", "image_path", "source_path", "input_file", "img_path"),
        "output_path": ("output_path", "save_path", "output_file", "result_path", "target_path"),
        "scale": ("scale", "upscale", "upscale_factor", "scale_factor"),
        "style": ("style", "model_type", "model_style", "mode"),
        "denoise": ("denoise", "use_denoise", "enable_denoise", "denoise_enabled"),
        "denoise_strength": ("denoise_strength", "strength", "denoise_level"),
    }
    values = {"input_path": str(input_path), "output_path": str(output_path),
              "scale": scale, "style": style, "denoise": denoise,
              "denoise_strength": denoise_strength}
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    kwargs = {}
    for key, names in aliases.items():
        name = next((n for n in names if n in params), None)
        if name:
            kwargs[name] = values[key]
        elif accepts_kwargs:
            kwargs[key] = values[key]
    required = [n for n, p in params.items() if n != "self"
                and p.default is inspect.Parameter.empty
                and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
                and n not in kwargs]
    if required:
        raise RuntimeError(f"无法适配当前 enhance() 的必需参数：{required}。请检查 ai_engine.py。")

    result = method(**kwargs)
    if isinstance(result, (str, Path)) and Path(result).is_file():
        return Path(result)
    if isinstance(result, dict):
        for key in ("output_path", "result_path", "save_path", "path"):
            if result.get(key) and Path(result[key]).is_file():
                return Path(result[key])
    if output_path.is_file():
        return output_path
    raise RuntimeError("模型调用结束但未找到输出图片。请确认 enhance() 会保存到 output_path 或返回结果路径。")


def _create_task(token: str, input_path: Path, output_path: Path) -> None:
    with _tasks_lock:
        _tasks[token] = {
            "state": "pending",
            "input_path": str(input_path),
            "output_path": str(output_path),
            "created_at": time.time(),
        }


def _start_worker(token: str, input_path: Path, output_path: Path,
                  scale: int, style: str, denoise: bool, denoise_strength: int) -> None:
    engine = get_engine()
    thread = Thread(
        target=_task_worker,
        args=(token, engine, input_path, output_path, scale, style, denoise, denoise_strength),
        daemon=True,
    )
    thread.start()


def _task_worker(task_id: str, engine, input_path: Path, output_path: Path,
                 scale: int, style: str, denoise: bool, denoise_strength: int):
    try:
        with _tasks_lock:
            if _tasks.get(task_id, {}).get("state") == "cancelled":
                return
            _tasks[task_id]["state"] = "running"

        acquired = False
        while not acquired:
            acquired = _inference_lock.acquire(timeout=2)
            if not acquired:
                with _tasks_lock:
                    if _tasks.get(task_id, {}).get("state") == "cancelled":
                        return

        try:
            result_path = run_enhance(
                engine, input_path, output_path, scale, style, denoise, denoise_strength
            )

            with _tasks_lock:
                task = _tasks.get(task_id)
                if task and task["state"] == "running":
                    task.update({
                        "state": "completed",
                        "result_url": f"/api/result/{result_path.name}",
                        "filename": result_path.name,
                        "scale": scale,
                        "style": style,
                    })
        except Exception as exc:
            with _tasks_lock:
                task = _tasks.get(task_id)
                if task and task["state"] == "running":
                    task.update({
                        "state": "error",
                        "error": str(exc),
                    })
        finally:
            _inference_lock.release()
    except Exception:
        pass
    finally:
        with _tasks_lock:
            task = _tasks.get(task_id)
            if task and task["state"] == "running":
                task["state"] = "completed"
        input_path.unlink(missing_ok=True)


@app.get("/", response_class=HTMLResponse)
def index():
    path = WEB_DIR / "index.html"
    if not path.is_file():
        raise HTTPException(500, "缺少 web/index.html")
    return HTMLResponse(
        path.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "AI Image Enhancer"}


@app.post("/api/enhance")
def enhance(file: UploadFile = File(...), style: str = Form("realistic"),
            scale: int = Form(4), denoise: bool = Form(False),
            denoise_strength: int = Form(5)):
    _validate_params(style, scale, denoise_strength)

    with _tasks_lock:
        running = any(t["state"] == "running" for t in _tasks.values())
    if running:
        raise HTTPException(409, "当前有任务正在运行，请等待完成或先取消。")

    input_path = None
    try:
        token, input_path = _save_upload(file)
        output_path = OUTPUT_DIR / f"{token}_{style}_{scale}x.png"
        _create_task(token, input_path, output_path)
        _start_worker(token, input_path, output_path, scale, style, denoise, denoise_strength)
        return {"task_id": token, "state": "pending", "message": "任务已提交"}
    except HTTPException:
        if input_path is not None:
            input_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        if input_path is not None:
            input_path.unlink(missing_ok=True)
        raise HTTPException(500, f"图片增强失败：{exc}") from exc


@app.post("/api/enhance/batch")
def enhance_batch(files: list[UploadFile] = File(...), style: str = Form("realistic"),
                  scale: int = Form(4), denoise: bool = Form(False),
                  denoise_strength: int = Form(5)):
    _validate_params(style, scale, denoise_strength)
    if not files:
        raise HTTPException(400, "未接收到任何文件")
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(400, f"单次批量最多 {MAX_BATCH_FILES} 张图片")

    engine = get_engine()
    tasks = []
    for file in files:
        input_path = None
        try:
            token, input_path = _save_upload(file)
            output_path = OUTPUT_DIR / f"{token}_{style}_{scale}x.png"
            _create_task(token, input_path, output_path)
            _start_worker(token, input_path, output_path, scale, style, denoise, denoise_strength)
            tasks.append({"task_id": token, "filename": file.filename, "state": "pending"})
        except HTTPException as exc:
            if input_path is not None:
                input_path.unlink(missing_ok=True)
            tasks.append({"task_id": None, "filename": file.filename,
                          "state": "error", "error": str(exc.detail)})
        except Exception as exc:
            if input_path is not None:
                input_path.unlink(missing_ok=True)
            tasks.append({"task_id": None, "filename": file.filename,
                          "state": "error", "error": str(exc)})
    if all(t["state"] == "error" for t in tasks):
        raise HTTPException(400, "批量上传的全部文件均无法处理：" +
                            "；".join(t["error"] for t in tasks))
    return {"tasks": tasks, "message": f"已提交 {sum(1 for t in tasks if t['task_id'])} 个任务"}


@app.get("/api/task/{task_id}")
async def task_status(task_id: str):
    with _tasks_lock:
        task = _tasks.get(task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    return {
        "task_id": task_id,
        "state": task["state"],
        "result_url": task.get("result_url"),
        "filename": task.get("filename"),
        "scale": task.get("scale"),
        "style": task.get("style"),
        "error": task.get("error"),
    }


@app.post("/api/cancel")
def cancel():
    engine = get_engine()
    engine.request_cancel()
    with _tasks_lock:
        for task_id, task in _tasks.items():
            if task["state"] in ("pending", "running"):
                task["state"] = "cancelled"
    return {"success": True, "message": "已发送取消信号"}


@app.post("/api/restart")
def restart():
    global _engine
    with _tasks_lock:
        for task in _tasks.values():
            if task["state"] in ("pending", "running"):
                raise HTTPException(409, "当前有任务正在运行，请先取消后再重启。")
    with _engine_lock:
        if _engine is not None:
            try:
                _engine.clear()
            except Exception:
                pass
            _engine = None
    Thread(target=_init_engine_background, daemon=True).start()
    return {"success": True, "message": "引擎已重启，正在后台加载模型"}


@app.get("/api/result/{filename}")
def result(filename: str):
    if Path(filename).name != filename:
        raise HTTPException(400, "非法文件名")
    path = OUTPUT_DIR / filename
    if not path.is_file():
        raise HTTPException(404, "结果图片不存在")
    return FileResponse(path, media_type="image/png", filename=filename)
