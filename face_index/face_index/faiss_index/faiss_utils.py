import faiss
import numpy as np

from face_index.logger import logger


class Faiss:

    def __init__(self, file_prefix: str = "", dim: int = 512):
        self.file_name = file_prefix
        self.index = faiss.IndexFlatL2(dim)

    def _write_index(self):
        return

    def _read_index(self):
        return

    def init_index(self, points: list):
        self.index.add(points)

    def add_point(self, point: np.array):
        self.index.add(point)

    def add_points(self, points: list):
        self.index.add(points)

    def rebuild_index(self, points: list):
        self.index.reset()
        self.index.add(points)

    def get_points_num(self):
        return self.index.ntotal

    def search(self, points: list, topn: int = 100):
        ntotal = self.index.ntotal
        if topn > ntotal:
            topn = ntotal
            logger.warning(f"topn parameter was limited by {ntotal}")

        return self.index.search(points, topn)
