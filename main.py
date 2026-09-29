"""Main FastAPI Application Entrypoint for SonicSentinel AI.
Enterprise Acoustic Intelligence Platform strictly complying with Aptech Technical Specifications.
"""
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.config import STATIC_DIR, DATA_DIR
from src.database.session import init_db
from src.api.routes import router as api_router
from src.api.websocket import ws_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle event handler: initializes SQLite database and seeds default categories."""
    print("[SonicSentinel AI] Initializing Autonomous Acoustic Intelligence System...")
    await init_db()
    print("[SonicSentinel AI] Database schemas verified and mandatory classes initialized.")
    yield
    print("[SonicSentinel AI] System shutdown complete.")


app = FastAPI(
    title="SonicSentinel AI - Acoustic Intelligence Platform",
    description="Enterprise Acoustic Intelligence, GCC-PHAT Radar Direction Finding, Dual-Model Consensus Arbitration, and Continuous Transfer Learning.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount REST API
app.include_router(api_router)

# Mount Static Assets
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_landing():
    """Serves the Landing Page (Introduction & Login)."""
    index_path = STATIC_DIR / "index.html"
    return FileResponse(str(index_path))

@app.get("/dashboard")
async def serve_dashboard():
    """Serves the primary Cyber-Security Acoustic Intelligence Dashboard."""
    dash_path = STATIC_DIR / "dashboard.html"
    return FileResponse(str(dash_path))


@app.get("/manifest.json")
async def serve_manifest():
    """Serves Progressive Web App manifest for desktop & mobile installation."""
    manifest_path = STATIC_DIR / "manifest.json"
    return FileResponse(str(manifest_path), media_type="application/manifest+json")


@app.get("/sw.js")
async def serve_service_worker():
    """Serves root-scoped Service Worker for offline capability & background notifications."""
    sw_path = STATIC_DIR / "sw.js"
    return FileResponse(str(sw_path), media_type="application/javascript")


@app.websocket("/ws/live-audio")
async def websocket_live_audio(websocket: WebSocket):
    """Real-time live monitoring WebSocket endpoint for continuous radar telemetry."""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep-alive heartbeat & bidirectional commands
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text('{"type": "pong"}')
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
