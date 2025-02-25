import pickle
from typing import List
import numpy as np
from sqlalchemy import select

from face_index import db_session_creator
from face_index.faiss_index.faiss_utils import Faiss
from face_index.logger import logger

from pg_model.meta import PortfolioEmbeddingInfo


class Manager:

    def __init__(self, images_index_map: List[str]) -> None:
        self.logger = logger
        self.faiss = None
        self.is_build = False
        self.images_index_map = images_index_map

    def init_index(self):
        recovered_points = self._await_db()
        self.faiss = Faiss()

        if len(recovered_points) > 0:
            self.faiss.init_index(recovered_points)
            self.is_build = True

        return True

    def clear_db(self):
        db_session = db_session_creator.get_new_session()
        try:
            db_session.query(PortfolioEmbeddingInfo).delete()
            db_session.commit()
            db_session.close()
            self.init_index()
            logger.info("Table face_embedding_info was cleared")
        except (Exception, ):
            db_session.rollback()
            db_session.close()

    def find_similar_images(self, embeddings: List[np.ndarray], topn: int,
                              threshold: float, expected_code: int = 200,
                              uuid: str = None) -> dict:

        results = []

        if len(self.images_index_map) == 0:
            self.logger.warning("Warning: No data in face index (in faiss).")
            return results

        embeddings = np.array(embeddings)

        D, I = self.faiss.search(embeddings, topn=topn)

        for indexes, distances in zip(I, D):
            result = []
            for idx, dist in zip(indexes, distances):
                if idx >= len(self.images_index_map):
                    self.logger.warning("Warning: Mismatched database index found. You should rebuild face index.")
                elif dist <= threshold:
                    neighbour_id = self.images_index_map[idx]
                    if uuid is not None and neighbour_id != uuid:
                        result.append(neighbour_id)

                    self.logger.info(f"Similar face {results} were found in the index.")

            results.append(result)

        return results

    def get_embeddings_by_uuid(self, uuid: str = None) -> List[List[float]]:

        embeddings = []

        if uuid is None:
            return embeddings

        db_session = db_session_creator.get_new_session()

        face_embeddings = db_session.query(PortfolioEmbeddingInfo) \
            .filter(PortfolioEmbeddingInfo.portfolio_id == uuid).all()

        for row in face_embeddings:
            embeddings.append(pickle.loads(row.vector))

        self.logger.info(f"{len(embeddings)} were found (portfolio_uuid={uuid})")
        db_session.commit()
        db_session.close()
        return embeddings

    def _await_db(self):
        db_exists = False
        recovered_points = []
        db_session = db_session_creator.get_new_session()
        while not db_exists:
            try:
                recovered_points = self._get_recovered_points(db_session)
                db_exists = True
            except (Exception,):
                pass
            db_session.commit()
            db_session.close()
            self.logger.info(f"Get all rows cnt={len(recovered_points)} from DB.")
            return recovered_points

    def _get_recovered_points(self, db_session):
        recovered_points = []

        query = select([PortfolioEmbeddingInfo]).order_by(PortfolioEmbeddingInfo.id.asc())
        proxy = db_session.execute(query)

        empty = False

        while not empty:
            batch = proxy.fetchmany(10000)

            if not batch:
                empty = True

            for row in batch:
                recovered_points.append(pickle.loads(row.vector))
                self.images_index_map.append(row.portfolio_id)

        return np.array(recovered_points)

    def insert_embeddings(self, uuid, embeddings: List[np.ndarray]):
        full_build_flag = False
        db_session = db_session_creator.get_new_session()
        for i in range(len(embeddings)):
            if embeddings[i] is not None:
                portfolio_count = db_session.query(PortfolioEmbeddingInfo) \
                    .filter(PortfolioEmbeddingInfo.portfolio_id == uuid) \
                    .count()
                if portfolio_count == 0:
                    to_insert = {"portfolio_id": uuid, "vector": pickle.dumps(embeddings[i])}
                    row_to_insert = PortfolioEmbeddingInfo(**to_insert)
                    db_session.add(row_to_insert)
                    self.faiss.add_point(np.array([embeddings[i]]))
                    self.images_index_map.append(uuid)
                else:
                    full_build_flag = True
                    db_session.query(PortfolioEmbeddingInfo) \
                        .filter(PortfolioEmbeddingInfo.portfolio_id == uuid) \
                        .update({"vector": pickle.dumps(embeddings[i])})
                self.logger.info(f"Face embedding added into db (portfolio_uuid={uuid})")
        db_session.commit()
        db_session.close()

        return full_build_flag
