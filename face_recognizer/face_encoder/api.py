import json
import logging
import os
import sys
from typing import List

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response
from face_encoder.exceptions.base_exception import ServiceBaseException
from face_encoder.logger import logger
from face_encoder.manager import Manager

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(asctime)s - %(pathname)s - %(levelname)s - %(message)s")

app = FastAPI()
manager = Manager()


@app.post("/insert_embeddings_to_db")
async def insert_embeddings_to_db(request: Request) -> None:
    data = await request.json()

    try:
        logger.info("Into api-method /insert_embeddings_to_db")
        config = data.get("config", {})

        s3_path = data.get("s3_path")

        if s3_path is None:
            return

        if not s3_path.endswith(manager.image_extensions):
            return

        logger.info(f"Get tdm {s3_path} with parameters {config}")
        embeddings, face_bbs = manager.encode_faces_from_s3_image(s3_path)

        uuid = data.get("uuid")

        manager.insert_embeddings_to_db(uuid, embeddings)

        return

    except ServiceBaseException as ex:
        logger.error(ex)
        raise HTTPException(status_code=ex.code, detail=str(ex.msg))
    except Exception as ex:
        logger.error(ex)
        raise HTTPException(status_code=500, detail=str(ex))


@app.post("/find_similar_images")
async def find_similar_images(request: Request) -> List[List[float]]:
    data = await request.json()

    try:
        logger.info("Into api-method /find_similar_images")
        config = data.get("config", {})
        topn = config.get("topn", 100)
        threshold = config.get("threshold", 0.7)
        s3_path =  data.get("s3_path")

        if s3_path is None:
            return

        if not s3_path.endswith(manager.image_extensions):
            return

        logger.info(f"Get tdm {s3_path} with parameters {config}")
        embeddings, face_bbs = manager.encode_faces_from_s3_image(s3_path)

        similar_images_ids = [[]]

        if not embeddings:
            logger.warning("No faces found in photo")
        else:
            uuid = data.get("uuid")
            similar_images = manager.find_similar_images(embeddings=embeddings,
                                                            topn=topn,
                                                            threshold=threshold,
                                                            uuid=uuid)

            similar_images_ids = similar_images["similar_images_ids"]


        return similar_images_ids

    except ServiceBaseException as ex:
        logger.error(ex)
        raise HTTPException(status_code=ex.code, detail=str(ex.msg))
    except Exception as ex:
        logger.error(ex)
        raise HTTPException(status_code=500, detail=str(ex))


@app.get("/", status_code=200)
async def get_info() -> Response:
    return "200"


def run_api() -> None:
    PORT = os.getenv("FACE_RECOGNIZER_PORT", "4446")
    uvicorn.run(app=app, host="0.0.0.0", port=int(PORT))  # noqa
