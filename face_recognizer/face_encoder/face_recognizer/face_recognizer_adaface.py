import logging
from typing import List, Tuple

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from adaface.models.ir_se_model import IR_SE_50

from face_encoder.face_recognizer.face_recognizer import FaceRecognizer


class FaceRecognizerAdaFace(FaceRecognizer):
    def __init__(self, weights_path="./weights/adaface_ir18_ms1mv3_ema.pth", on_gpu=False) -> None:
        super().__init__()
        self.logger = logging.getLogger()
        self._set_device(on_gpu=on_gpu)

        self.model = IR_SE_50(num_classes=0, use_dropout=False)
        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict["state_dict"])
        self.model.eval()
        self.model.to(self.device)

        self.face_detector = MTCNN(
            image_size=160,
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
            extended_bb = self.__extend_rectangle(face_bb, img_rgb, 0.1, 0.1)
            face_img = img_rgb[int(extended_bb[1]):int(extended_bb[3]), int(extended_bb[0]):int(extended_bb[2])]
            resized_face = cv2.resize(face_img, (112, 112))  # Размеры входа для AdaFace

            input_tensor = torch.from_numpy(resized_face.transpose((2, 0, 1))).unsqueeze(0).to(dtype=torch.float32,
                                                                                               device=self.device)

            with torch.no_grad():
                output = self.model(input_tensor)

            emb = output.cpu().detach().numpy()[0]
            normed_emb = emb / np.linalg.norm(emb)

            embeddings.append(normed_emb.tolist())

        return embeddings, face_bbs

    def __find_faces(self, _img_rgb: np.ndarray) -> Tuple[List[List], np.ndarray]:
        bounding_boxes, _, _ = self.face_detector.detect(_img_rgb, landmarks=True)
        face_bbs = []
        if bounding_boxes is not None:
            for face in bounding_boxes:
                x_min, y_min, x_max, y_max = face[:4]
                face_bbs.append([x_min, y_min, x_max, y_max])
        return face_bbs, _img_rgb

    def __calculate_embeddings(self, _img_rgb: np.ndarray) -> Tuple[List[np.ndarray], List[List[float]]]:
        face_bbs, _ = self.__find_faces(_img_rgb)
        if len(face_bbs) == 0:
            return [], []
        return self.__calculate_embedding_from_face(face_bbs, _img_rgb)

    def __extend_rectangle(self, rectangle: List[float], frame: np.ndarray, extend_x: float = 0.1,
                           extend_y: float = 0.1) -> Tuple:
        ax, ay, bx, by = rectangle
        w, h = bx - ax, by - ay
        dx, dy = w * extend_x, h * extend_y
        return (
            max(0, ax - dx),
            max(0, ay - dy),
            min(frame.shape[1], bx + dx),
            min(frame.shape[0], by + dy)
        )

    def get_image_embeddings(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[List[float]]]:
        return self.__calculate_embeddings(image)