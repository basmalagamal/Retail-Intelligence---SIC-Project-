from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from . import churn
from .recsys import RecSys

STATIC = Path(__file__).resolve().parent.parent / "static"
state = {}


@asynccontextmanager
async def lifespan(app):
    state["rs"] = RecSys()
    try:
        churn.state["svc"] = churn.ChurnService(state["rs"])
    except Exception as e:
        churn.state["error"] = f"{type(e).__name__}: {e}"
        print("[churn] disabled:", churn.state["error"])
    yield


app = FastAPI(title="H&M Recommender", lifespan=lifespan)
app.include_router(churn.router)


def rs():
    return state["rs"]


@app.get("/api/health")
def health():
    r = rs()
    return {"articles": len(r.ids), "categories": len(r.categories()),
            "with_images": sum(r.image_path(a) is not None for a in r.ids[:500]), "image_sample": 500}


@app.get("/api/categories")
def categories():
    return rs().categories()


@app.get("/api/bestsellers")
def bestsellers(category: str | None = None, limit: int = Query(12, ge=1, le=60)):
    return rs().bestsellers(category, limit)


@app.get("/api/bestsellers/by-category")
def bestsellers_by_category(limit: int = Query(8, ge=1, le=30)):
    return [{"category": c["name"], "items": rs().bestsellers(c["name"], limit)} for c in rs().categories()]


@app.get("/api/product/{article_id}")
def product(article_id: str, k: int = Query(12, ge=1, le=48)):
    r = rs()
    if article_id not in r.id2row:
        raise HTTPException(404, "unknown article_id")
    return {"item": r.item(r.id2row[article_id]), "similar": r.similar(article_id, k)}


@app.post("/api/search-by-image")
async def search_by_image(file: UploadFile = File(...), k: int = Query(12, ge=1, le=48)): #k = Number of similar products to return for the uploaded image.
    data = await file.read()
    try:
        return {"results": rs().similar_to_image(data, k)}
    except OSError:
        raise HTTPException(400, "could not read that file as an image")


@app.get("/api/image/{article_id}")
def image(article_id: str):
    p = rs().image_path(article_id) if article_id in rs().id2row else None
    if p:
        return FileResponse(p, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="256" height="320"><rect width="100%" height="100%" '
           'fill="#eee"/><text x="50%" y="50%" fill="#999" font-family="sans-serif" font-size="14" '
           'text-anchor="middle">no image</text></svg>')
    return Response(svg, media_type="image/svg+xml")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
