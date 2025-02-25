import logging
import os
import sys
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from face_index.manager import Manager

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s - %(pathname)s - %(levelname)s - %(message)s")

app = FastAPI()
logger = logging.getLogger()
manager = Manager([])


@app.post("/init_index")
async def init_index(request: Request):
    try:
        manager.init_index()
        return JSONResponse({"result": "True"})
    except Exception as ex:
        logger.error(ex)
        raise HTTPException(status_code=500, detail=str(ex))


@app.post("/insert_embeddings_to_db")
async def insert_embeddings_db(request: Request):
    data = await request.json()

    try:
        uuid = data.get("uuid")
        embeddings = data.get("embeddings")
        manager.insert_embeddings(uuid, embeddings)
        return JSONResponse({"result": True})
    except Exception as ex:
        logger.error(ex)
        raise HTTPException(status_code=500, detail=str(ex))


@app.post("/find_similar_images")
async def find_similar_images(request: Request):
    data = await request.json()

    try:
        embeddings = data.get("embeddings")
        topn = data.get("topn", 100)
        threshold = data.get("threshold", 0.7)
        uuid = data.get("uuid", None)
        similar_images_ids = manager.find_similar_images(embeddings=embeddings, topn=topn, threshold=threshold, uuid=uuid)
        return JSONResponse({"similar_images_ids": similar_images_ids})
    except Exception as ex:
        logger.error(ex)
        raise HTTPException(status_code=500, detail=str(ex))


@app.post("/get_embeddings_by_uuid")
async def get_embeddings_by_uuid(request: Request):
    data = await request.json()

    try:
        uuid = data.get("uuid", None)
        embeddings = manager.get_embeddings_by_uuid(uuid)
        return JSONResponse({"embeddings": embeddings})
    except Exception as ex:
        logger.error(ex)
        raise HTTPException(status_code=500, detail=str(ex))


@app.get("/clear_db")
async def clear_db(request: Request):
    try:
        manager.clear_db()
        return JSONResponse({"result": True})
    except Exception as ex:
        logger.error(ex)
        raise HTTPException(status_code=500, detail=str(ex))


def run_api() -> None:
    manager.init_index()
    PORT = os.getenv("FACE_INDEX_PORT", "4445")
    uvicorn.run(app=app, host="0.0.0.0", port=int(PORT))  # noqa
