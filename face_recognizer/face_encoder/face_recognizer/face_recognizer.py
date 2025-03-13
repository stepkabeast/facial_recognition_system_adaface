import abc
from typing import Tuple, List

import numpy as np


class FaceRecognizer(abc.ABC):

    @abc.abstractmethod
    def get_image_embeddings(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[List[float]]]:
        pass