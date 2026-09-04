from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .routes import router

app = FastAPI(
    title="Voice Integrity Verification API",
    description="Real-Time AI Voice Cloning & Deepfake Impersonation Detection",
    version="1.0.0"
)

# Open CORS so web clients can talk to us without getting blocked by browser policies
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

# Serve the React frontend (frontend/dist) or fallback to static prototype
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
STATIC_DIR = Path(__file__).resolve().parent / "static"

if FRONTEND_DIST.exists() and (FRONTEND_DIST / "index.html").exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str = ""):
        # Don't intercept API routes or documentation
        if full_path.startswith("api/") or full_path in ["docs", "redoc", "openapi.json"]:
            return None
        candidate_file = FRONTEND_DIST / full_path
        if full_path and candidate_file.exists() and candidate_file.is_file():
            return FileResponse(candidate_file)
        return FileResponse(FRONTEND_DIST / "index.html")

elif STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_ui():
        # Serve the single-page dashboard app
        return FileResponse(STATIC_DIR / "index.html")


def start_server(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
