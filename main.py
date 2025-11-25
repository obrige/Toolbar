#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import base64
import time
import random
import string
import re
import logging
import httpx  # 使用 httpx 替代 requests 以实现真正的异步
from typing import List, Dict, Generator, Optional, AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


# --- 配置日志 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --- Pydantic 模型 ---
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "gemini-2.5-flash"
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    session_id: Optional[str] = ""


class ChatCompletionChoice(BaseModel):
    index: int
    message: Optional[Dict] = None
    delta: Optional[Dict] = None
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[Dict] = None


# --- 核心：Toolbaz API 适配器 ---
class ToolbazToOpenAI:
    def __init__(self):
        self.base_url = "https://data.toolbaz.com"
        # 使用 httpx.AsyncClient 替代 requests.Session 以实现真正的异步
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(120.0, connect=10.0),  # 增加了超时时间
            headers={
                "Accept": "*/*",
                "Accept-Encoding": "gzip, deflate, br, zstd",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6",
                "Connection": "keep-alive",
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                "Origin": "https://toolbaz.com",
                "Referer": "https://toolbaz.com/",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-site",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36 Edg/142.0.0.0",
                "sec-ch-ua": '"Chromium";v="142", "Microsoft Edge";v="142", "Not_A Brand";v="99"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": '"Windows"'
            }
        )
        self.last_token = None

    @staticmethod
    def generate_random_string(length: int = 32) -> str:
        chars = string.ascii_letters + string.digits
        return ''.join(random.choice(chars) for _ in range(length))

    def generate_device_token(self, use_last_token: bool = False) -> str:
        if use_last_token and self.last_token:
            return self.last_token

        device_info = {
            "bUtGb": {
                "nV5kP": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36 Edg/142.0.0.0",
                "lQ9jX": "zh-CN",
                "sD2zR": "1920x1080",
                "tY4hL": "Asia/Shanghai",
                "pL8mC": "Win32",
                "cQ3vD": 24,
                "hK7jN": 20
            },
            "uT4bX": {"mM9wZ": [], "kP8jY": []},
            "tuTcS": int(time.time()),
            "tDfxy": "null",
            "RtyJt": self.generate_random_string(32)
        }

        json_str = json.dumps(device_info, separators=(',', ':'))
        base64_encoded = base64.b64encode(json_str.encode('utf-8')).decode('utf-8')
        token = f"XFriIz{base64_encoded}"
        return token

    async def get_capcha_token(self, session_id: str = "", use_last_token: bool = False) -> Optional[str]:
        url = f"{self.base_url}/token.php"
        device_token = self.generate_device_token(use_last_token)
        data = {"session_id": session_id, "token": device_token}
        try:
            response = await self._client.post(url, data=data)
            response.raise_for_status()
            result = response.json()
            if "token" in result:
                self.last_token = device_token
                return result["token"]
            return None
        except Exception as e:
            logger.error(f"获取 capcha token 失败: {e}")
            return None

    def messages_to_toolbaz_format(self, messages: List[ChatMessage]) -> str:
        conversation_parts = []
        for msg in messages:
            if msg.role == "system":
                conversation_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                conversation_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                conversation_parts.append(f"Assistant: {msg.content}")
        return "\n".join(conversation_parts)

    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        formatted_message = self.messages_to_toolbaz_format(request.messages)
        capcha_token = await self.get_capcha_token(request.session_id, len(request.messages) > 1)
        if not capcha_token:
            raise HTTPException(status_code=500, detail="Failed to get capcha token")

        url = f"{self.base_url}/writing.php"
        toolbaz_message = f"ㅤ : {formatted_message}ㅤ"
        data = {
            "text": toolbaz_message,
            "capcha": capcha_token,
            "model": request.model,
            "session_id": request.session_id,
        }
        # 尝试传递更多参数，即使后端可能不使用
        if request.temperature is not None:
            data["temperature"] = request.temperature
        if request.max_tokens is not None:
            data["max_tokens"] = request.max_tokens
        
        try:
            response = await self._client.post(url, data=data)
            response.raise_for_status()
            
            # --- 核心修复：过滤掉后端返回的 [model: ...] 标签 ---
            cleaned_content = re.sub(r'^\[model: .+?\]\s*', '', response.text, flags=re.MULTILINE)
            
            completion_id = f"chatcmpl-{self.generate_random_string(29)}"
            created = int(time.time())
            
            choice = ChatCompletionChoice(
                index=0,
                message={"role": "assistant", "content": cleaned_content},
                finish_reason="stop"
            )
            
            return ChatCompletionResponse(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[choice]
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"Toolbaz API 请求失败: {e.response.status_code} - {e.response.text}")
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except Exception as e:
            logger.error(f"chat_completion 发生未知错误: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def chat_completion_stream(self, request: ChatCompletionRequest) -> AsyncGenerator[str, None]:
        formatted_message = self.messages_to_toolbaz_format(request.messages)
        capcha_token = await self.get_capcha_token(request.session_id, len(request.messages) > 1)
        if not capcha_token:
            # 在流中处理错误，通过 SSE 发送错误信息
            error_data = {"error": {"message": "Failed to get capcha token", "type": "internal_error"}}
            yield f"data: {json.dumps(error_data)}\n\n"
            return

        url = f"{self.base_url}/writing.php"
        toolbaz_message = f"ㅤ : {formatted_message}ㅤ"
        data = {
            "text": toolbaz_message,
            "capcha": capcha_token,
            "model": request.model,
            "session_id": request.session_id,
        }
        if request.temperature is not None:
            data["temperature"] = request.temperature
        if request.max_tokens is not None:
            data["max_tokens"] = request.max_tokens

        completion_id = f"chatcmpl-{self.generate_random_string(29)}"
        created = int(time.time())
        
        try:
            async with self._client.stream("POST", url, data=data) as resp:
                resp.raise_for_status()
                
                # 发送初始的 "role" delta
                start_chunk = {
                    "id": completion_id, "object": "chat.completion.chunk", "created": created,
                    "model": request.model, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(start_chunk)}\n\n"
                
                # --- 核心修复：流式过滤逻辑 ---
                header_stripped = False
                initial_buffer = ""
                
                # 使用 aiter_text 并增大 chunk_size，提高效率
                async for chunk in resp.aiter_text(chunk_size=128):
                    if not header_stripped:
                        initial_buffer += chunk
                        # 检查缓冲区中是否包含完整的模型标签
                        if re.search(r'^\[model: .+?\]', initial_buffer, flags=re.MULTILINE):
                            # 移除标签和可能的空白
                            cleaned_chunk = re.sub(r'^\[model: .+?\]\s*', '', initial_buffer, flags=re.MULTILINE)
                            if cleaned_chunk:
                                delta_chunk = {
                                    "id": completion_id, "object": "chat.completion.chunk", "created": created,
                                    "model": request.model, "choices": [{"index": 0, "delta": {"content": cleaned_chunk}, "finish_reason": None}]
                                }
                                yield f"data: {json.dumps(delta_chunk)}\n\n"
                            header_stripped = True
                    else:
                        # 头部已处理，直接流式传输后续内容
                        delta_chunk = {
                            "id": completion_id, "object": "chat.completion.chunk", "created": created,
                            "model": request.model, "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}]
                        }
                        yield f"data: {json.dumps(delta_chunk)}\n\n"
                
                # 发送结束标记
                end_chunk = {
                    "id": completion_id, "object": "chat.completion.chunk", "created": created,
                    "model": request.model, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }
                yield f"data: {json.dumps(end_chunk)}\n\n"
                yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"chat_completion_stream 发生错误: {e}")
            error_data = {"error": {"message": str(e), "type": "internal_error"}}
            yield f"data: {json.dumps(error_data)}\n\n"

    async def close(self):
        """关闭异步客户端，释放资源。"""
        await self._client.aclose()


# --- FastAPI 应用实例 ---
app = FastAPI(title="Toolbaz to OpenAI API (Optimized)", version="2.0.0")
toolbaz_client = ToolbazToOpenAI()


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    logger.info(f"收到请求: model={request.model}, stream={request.stream}, messages={len(request.messages)}")
    if request.stream:
        # --- 核心修复：使用正确的 media_type ---
        return StreamingResponse(
            toolbaz_client.chat_completion_stream(request),
            media_type="text/event-stream"
        )
    else:
        return await toolbaz_client.chat_completion(request)


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "gemini-2.5-flash", "object": "model", "created": int(time.time())},
            {"id": "gemini-2.5-pro", "object": "model", "created": int(time.time())},
            {"id": "claude-sonnet-4", "object": "model", "created": int(time.time())},
            {"id": "gpt-5", "object": "model", "created": int(time.time())},
            {"id": "grok-4-fast", "object": "model", "created": int(time.time())}
        ]
    }


@app.get("/health")
async def health():
    """健康检查端点。"""
    return {"status": "ok"}


@app.on_event("shutdown")
async def on_shutdown():
    """应用关闭时，清理资源。"""
    logger.info("正在关闭应用，清理资源...")
    await toolbaz_client.close()
    logger.info("资源已清理。")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
