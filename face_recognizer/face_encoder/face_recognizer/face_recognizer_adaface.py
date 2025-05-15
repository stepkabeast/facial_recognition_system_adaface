import logging
from typing import List, Tuple

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from torch.nn import functional as F

from head import AdaFace

from face_encoder.face_recognizer.face_recognizer import FaceRecognizer


class FaceRecognizerAdaFace(FaceRecognizer):
    def __init__(self, weights_path="./weights/adaface_ir101_ms1mv3.ckpt", on_gpu=False) -> None:
        super().__init__()
        self.logger = logging.getLogger()
        self._set_device(on_gpu=on_gpu)

        self.head = AdaFace(embedding_size=512, classnum=70722, m=0.4, h=0.333, s=64., t_alpha=1.0)
        state_dict = torch.load(weights_path, map_location=self.device)
        self.head.load_state_dict(state_dict["state_dict"], strict=False)
        self.head.eval()
        self.head.to(self.device)

        self.face_detector = MTCNN(
            image_size=112,  # Input size for AdaFace should be 112x112
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            device=self.device
        )

    def _set_device(self, on_gpu: bool) -> None:
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.logger.info("Используется GPU.")
        else:
            self.device = torch.device("cpu")
            self.logger.info("Используется CPU.")

    def __calculate_embedding_from_face(self, face_bbs: List[List[float]], img_rgb: List[np.ndarray]) -> Tuple[
        List[np.ndarray], List[List]]:
        embeddings = []
        for face_bb in face_bbs:
            x1, y1, x2, y2 = list(map(int, face_bb))
            face_img = img_rgb[y1:y2, x1:x2]
            resized_face = cv2.resize(face_img, (112, 112))  # Resize to match expected input shape
            input_tensor = torch.tensor(resized_face, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self.head(input_tensor)[0][0]  # Extract only the first part of the output
                norm = torch.norm(embedding, dim=-1, keepdim=True)
                normalized_embedding = embedding / norm

            embeddings.append(normalized_embedding.cpu().numpy())

        return embeddings, face_bbs

    def __find_faces(self, _img_rgb: np.ndarray) -> Tuple[List[List], np.ndarray]:
        boxes, _, _ = self.face_detector.detect(_img_rgb, landmarks=True)
        if boxes is None or len(boxes) == 0:
            return [], _img_rgb
        return boxes, _img_rgb

    def get_image_embeddings(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[List[float]]]:
        detected_faces, _ = self.__find_faces(image)
        if len(detected_faces) == 0:
            return [], []
        return self.__calculate_embedding_from_face(detected_faces, image)