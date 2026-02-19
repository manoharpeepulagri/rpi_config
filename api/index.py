from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

try:
    import cantools
except Exception:
    cantools = None

BASE_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

app = FastAPI(title="Nandi RPi Config (Vercel)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR)) if TEMPLATES_DIR.exists() else None


def _bucket_or_raise() -> str:
    bucket = os.getenv("S3_BUCKET_NAME", "").strip()
    if not bucket:
        raise ValueError("S3_BUCKET_NAME is required.")
    return bucket


def _region() -> str:
    return os.getenv("AWS_REGION", "ap-south-1").strip() or "ap-south-1"


@lru_cache(maxsize=1)
def _s3_client():
    kwargs: dict[str, Any] = {"region_name": _region()}
    access = os.getenv("AWS_ACCESS_KEY_ID", "").strip()
    secret = os.getenv("AWS_SECRET_ACCESS_KEY", "").strip()
    token = os.getenv("AWS_SESSION_TOKEN", "").strip()
    if access and secret:
        kwargs["aws_access_key_id"] = access
        kwargs["aws_secret_access_key"] = secret
        if token:
            kwargs["aws_session_token"] = token
    return boto3.client("s3", **kwargs)


def _normalize_device(device: str) -> str:
    value = (device or "").strip().strip("/")
    if not value:
        raise ValueError("Device is required.")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise ValueError("Device contains invalid characters.")
    return value


def _normalize_file(file_name: str) -> str:
    value = (file_name or "").strip()
    if not value:
        raise ValueError("File is required.")
    if value.startswith("/") or value.endswith("/"):
        raise ValueError("Invalid file path.")
    if ".." in value or "\\" in value:
        raise ValueError("Unsafe file path.")
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", value):
        raise ValueError("File contains invalid characters.")
    return value


def _device_prefix(device: str) -> str:
    return f"devices/{device}/"


def _config_key(device: str, file_name: str) -> str:
    return f"devices/{device}/{file_name}"


def _max_files() -> int:
    raw = os.getenv("RPI_MAX_FILES", "300").strip()
    try:
        return max(50, min(2000, int(raw)))
    except ValueError:
        return 300


def _list_devices() -> list[str]:
    bucket = _bucket_or_raise()
    paginator = _s3_client().get_paginator("list_objects_v2")
    devices: set[str] = set()
    for page in paginator.paginate(Bucket=bucket, Prefix="devices/", Delimiter="/"):
        for item in page.get("CommonPrefixes", []):
            prefix = str(item.get("Prefix", ""))
            parts = prefix.strip("/").split("/")
            if len(parts) >= 2 and parts[0] == "devices":
                devices.add(parts[1])
    return sorted(devices)


def _list_files(device: str) -> tuple[list[dict[str, Any]], bool]:
    bucket = _bucket_or_raise()
    prefix = _device_prefix(device)
    paginator = _s3_client().get_paginator("list_objects_v2")
    files: list[dict[str, Any]] = []
    truncated = False
    cap = _max_files()

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for obj in page.get("Contents", []):
            key = str(obj.get("Key", ""))
            if not key.startswith(prefix):
                continue
            rel = key[len(prefix):].strip()
            if not rel or rel.endswith("/") or "/" in rel:
                continue
            files.append(
                {
                    "name": rel,
                    "size": int(obj.get("Size") or 0),
                    "last_modified": (
                        obj.get("LastModified").isoformat()
                        if isinstance(obj.get("LastModified"), datetime)
                        else None
                    ),
                }
            )
            if len(files) >= cap:
                truncated = True
                break
        if truncated:
            break

    files.sort(key=lambda item: item["name"].lower())
    return files, truncated


class ConfigWriteRequest(BaseModel):
    device: str = Field(min_length=1, max_length=200)
    file: str = Field(min_length=1, max_length=300)
    content: str = Field(default="")


class DbcParseRequest(BaseModel):
    content: str = Field(default="")


class ActionRequest(BaseModel):
    device: str = Field(min_length=1, max_length=200)
    action: str = Field(min_length=1, max_length=40)
    auto_reset_seconds: int = Field(default=10, ge=0, le=1800)


def _write_action_file(device: str, action: str) -> None:
    key = _config_key(device, "device_control.json")
    body = {"action": action}
    _s3_client().put_object(
        Bucket=_bucket_or_raise(),
        Key=key,
        Body=json.dumps(body).encode("utf-8"),
        ContentType="application/json",
    )


async def _reset_action_after_delay(device: str, delay_seconds: int) -> None:
    if delay_seconds <= 0:
        return
    await asyncio.sleep(delay_seconds)
    try:
        _write_action_file(device, "none")
    except Exception:
        pass


def _render_rpi_page(request: Request) -> HTMLResponse:
    if templates is None:
        return HTMLResponse(
            "<h3 style='font-family:sans-serif'>Template directory not found in deployment bundle.</h3>",
            status_code=500,
        )
    try:
        return templates.TemplateResponse("rpi_config.html", {"request": request})
    except Exception as exc:
        return HTMLResponse(
            f"<h3 style='font-family:sans-serif'>Template render failed: {exc}</h3>",
            status_code=500,
        )


@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    return _render_rpi_page(request)


@app.get("/rpi-config", response_class=HTMLResponse)
async def rpi_config_page(request: Request) -> HTMLResponse:
    return _render_rpi_page(request)


@app.get("/overview", response_class=HTMLResponse)
async def overview_placeholder() -> HTMLResponse:
    return HTMLResponse("<h3 style='font-family: sans-serif;'>Overview route is not part of this standalone app.</h3>")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "rpi-config-vercel"}


@app.get("/ops/rpi/devices")
async def ops_rpi_devices() -> JSONResponse:
    try:
        return JSONResponse(content={"devices": _list_devices()})
    except ValueError as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=400)
    except (BotoCoreError, ClientError) as exc:
        return JSONResponse(content={"error": f"S3 list failed: {exc}"}, status_code=500)


@app.get("/ops/rpi/files")
async def ops_rpi_files(device: str) -> JSONResponse:
    try:
        safe_device = _normalize_device(device)
        files, truncated = _list_files(safe_device)
        return JSONResponse(
            content={
                "device": safe_device,
                "files": files,
                "count": len(files),
                "truncated": truncated,
            }
        )
    except ValueError as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=400)
    except (BotoCoreError, ClientError) as exc:
        return JSONResponse(content={"error": f"S3 list failed: {exc}"}, status_code=500)


@app.get("/ops/rpi/config")
async def ops_rpi_get_config(device: str, file: str) -> JSONResponse:
    try:
        safe_device = _normalize_device(device)
        safe_file = _normalize_file(file)
        key = _config_key(safe_device, safe_file)
        obj = _s3_client().get_object(Bucket=_bucket_or_raise(), Key=key)
        body = obj["Body"].read().decode("utf-8", errors="replace")
        last_modified = obj.get("LastModified")
        return JSONResponse(
            content={
                "device": safe_device,
                "file": safe_file,
                "s3_key": key,
                "content": body,
                "last_modified": (
                    last_modified.isoformat() if isinstance(last_modified, datetime) else None
                ),
            }
        )
    except ValueError as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=400)
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in {"NoSuchKey", "404", "NotFound"}:
            return JSONResponse(content={"error": "Config file not found in S3."}, status_code=404)
        return JSONResponse(content={"error": f"S3 read failed: {exc}"}, status_code=500)
    except BotoCoreError as exc:
        return JSONResponse(content={"error": f"S3 read failed: {exc}"}, status_code=500)


@app.post("/ops/rpi/config")
async def ops_rpi_save_config(payload: ConfigWriteRequest) -> JSONResponse:
    try:
        safe_device = _normalize_device(payload.device)
        safe_file = _normalize_file(payload.file)
        key = _config_key(safe_device, safe_file)
        _s3_client().put_object(
            Bucket=_bucket_or_raise(),
            Key=key,
            Body=(payload.content or "").encode("utf-8"),
            ContentType="text/plain; charset=utf-8",
        )
        return JSONResponse(
            content={
                "status": "success",
                "device": safe_device,
                "file": safe_file,
                "s3_key": key,
            }
        )
    except ValueError as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=400)
    except (BotoCoreError, ClientError) as exc:
        return JSONResponse(content={"error": f"S3 write failed: {exc}"}, status_code=500)


@app.delete("/ops/rpi/config")
async def ops_rpi_delete_config(device: str, file: str) -> JSONResponse:
    try:
        safe_device = _normalize_device(device)
        safe_file = _normalize_file(file)
        key = _config_key(safe_device, safe_file)
        _s3_client().delete_object(Bucket=_bucket_or_raise(), Key=key)
        return JSONResponse(
            content={
                "status": "success",
                "device": safe_device,
                "file": safe_file,
                "s3_key": key,
            }
        )
    except ValueError as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=400)
    except (BotoCoreError, ClientError) as exc:
        return JSONResponse(content={"error": f"S3 delete failed: {exc}"}, status_code=500)


@app.post("/ops/rpi/dbc/parse")
async def ops_rpi_parse_dbc(payload: DbcParseRequest) -> JSONResponse:
    try:
        if cantools is None:
            raise ValueError("cantools is not installed.")
        content = payload.content or ""
        if not content.strip():
            raise ValueError("DBC content is empty.")

        db = cantools.database.load_string(content)
        messages: list[dict[str, Any]] = []
        signals_by_message: dict[str, list[dict[str, Any]]] = {}

        for message in db.messages:
            message_name = str(message.name)
            frame_format = "Extended" if bool(getattr(message, "is_extended_frame", False)) else "Standard"
            if bool(getattr(message, "is_fd", False)):
                frame_format = f"{frame_format} FD"

            messages.append(
                {
                    "name": message_name,
                    "id_decimal": int(getattr(message, "frame_id", 0)),
                    "id_hex": f"0x{int(getattr(message, 'frame_id', 0)):X}",
                    "frame_format": frame_format,
                    "brs": bool(getattr(message, "is_fd", False)),
                    "dlc": int(getattr(message, "length", 0)),
                    "tx_node": ", ".join(getattr(message, "senders", []) or []),
                    "comment": str(getattr(message, "comment", "") or ""),
                    "attributes": "",
                }
            )

            rows: list[dict[str, Any]] = []
            for signal in message.signals:
                rows.append(
                    {
                        "name": str(signal.name),
                        "type": "Float" if bool(getattr(signal, "is_float", False)) else (
                            "Signed" if bool(getattr(signal, "is_signed", False)) else "Unsigned"
                        ),
                        "byteorder": "Intel" if str(getattr(signal, "byte_order", "")) == "little_endian" else "Motorola",
                        "mode": "Signal",
                        "bitpos": int(getattr(signal, "start", 0)),
                        "length": int(getattr(signal, "length", 0)),
                        "factor": float(getattr(signal, "scale", 1)),
                        "offset": float(getattr(signal, "offset", 0)),
                        "minimum": getattr(signal, "minimum", None),
                        "maximum": getattr(signal, "maximum", None),
                        "unit": getattr(signal, "unit", None),
                        "comment": str(getattr(signal, "comment", "") or ""),
                        "values": "",
                    }
                )
            signals_by_message[message_name] = rows

        return JSONResponse(
            content={
                "status": "success",
                "message_count": len(messages),
                "signal_count": sum(len(v) for v in signals_by_message.values()),
                "messages": messages,
                "signals_by_message": signals_by_message,
            }
        )
    except ValueError as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse(content={"error": f"DBC parse failed: {exc}"}, status_code=500)


@app.post("/ops/rpi/action")
async def ops_rpi_action(payload: ActionRequest) -> JSONResponse:
    try:
        safe_device = _normalize_device(payload.device)
        action = payload.action.strip().lower()
        if action not in {"reboot", "poweroff", "none"}:
            raise ValueError(f"Unsupported action '{payload.action}'.")

        _write_action_file(safe_device, action)
        if action in {"reboot", "poweroff"} and payload.auto_reset_seconds > 0:
            asyncio.create_task(_reset_action_after_delay(safe_device, int(payload.auto_reset_seconds)))
        return JSONResponse(
            content={
                "status": "success",
                "device": safe_device,
                "action": action,
                "auto_reset_seconds": int(payload.auto_reset_seconds),
            }
        )
    except ValueError as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=400)
    except (BotoCoreError, ClientError) as exc:
        return JSONResponse(content={"error": f"S3 action update failed: {exc}"}, status_code=500)
