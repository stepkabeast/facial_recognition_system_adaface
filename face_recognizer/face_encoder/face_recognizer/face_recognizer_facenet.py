import logging
from typing import List, Tuple

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

from face_encoder.face_recognizer.face_recognizer import FaceRecognizer
import ssl
ssl.PROTOCOL_TLSv1_2 = ssl.TLSVersion.TLSv1_2

class FaceRecognizerFacenet(FaceRecognizer):
    def __init__(self, on_gpu=False) -> None:
        self.logger = logging.getLogger()
        self._set_device(on_gpu=on_gpu)
        # Create an inception resnet (in eval mode):
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()
        # create a face detection pipeline using MTCNN:
        self.face_detector = MTCNN(image_size=160,
                                   margin=0, min_face_size=20,
                                   thresholds=[0.6, 0.7, 0.7],
                                   factor=0.709,
                                   post_process=True,
                                   device=self.device)

    def _set_device(self, on_gpu: bool) -> None:
        """
        Set device configuration
        """
        if on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.location = lambda storage, loc: storage.cuda()
            self.logger.info("GPU is used")
        else:
            self.device = torch.device("cpu")
            self.location = 'cpu'
            self.logger.info("CPU is used")

    def __calculate_embedding_from_face(self, face_bbs: List[List[float]], img_rgb: List[np.ndarray]) -> \
            Tuple[List[np.ndarray], List[List]]:
        picture_embeddings = []
        for face_idx, face_bb in enumerate(face_bbs):
            face_bb = self.__extend_rectangle(face_bb, img_rgb, 1. / 0.9 - 1., 1. / 0.9 - 1.)
            face_rgb = img_rgb[face_bb[1]:face_bb[3], face_bb[0]:face_bb[2]]
            face_rgb_float = face_rgb.astype(np.float32)
            face_rgb_float_resize = cv2.resize(face_rgb_float, (160, 160))

            try:
                face_rgb_face, prob = self.face_detector(face_rgb_float_resize, return_prob=True)
            except(Exception,):
                self.logger.info("Can't calculate embedding from face")
                return picture_embeddings, face_bbs

            if face_rgb_face is None:
                continue

            face_rgb_face = face_rgb_face.unsqueeze(0)
            face_emb = self.resnet(face_rgb_face)
            face_emb = face_emb.detach().numpy()
            n = np.linalg.norm(face_emb, axis=1)
            face_emb = face_emb / n[:, np.newaxis]
            face_emb = np.array(face_emb).astype(np.float32)
            picture_embeddings += [face_emb[0].tolist()]

        return picture_embeddings, face_bbs

    def __find_faces(self, _img_rgb: np.ndarray) -> Tuple[List[List], np.ndarray]:
        bounding_boxes, conf, landmarks = self.face_detector.detect(_img_rgb, landmarks=True)
        face_bbs = []
        if bounding_boxes is None:
            return None, _img_rgb
        for face in bounding_boxes:
            width, height = _img_rgb.shape[1], _img_rgb.shape[0]
            det = [min(width, float(face[0])),
                   min(height, float(face[1])),
                   min(width, float(face[2])),
                   min(height, float(face[3]))]
            face_bbs += [det]
        return face_bbs, _img_rgb

    def __calculate_embeddings(self, _img_rgb: np.ndarray) -> Tuple[List[np.ndarray], List[List[float]]]:
        face_bbs, img_rgb = self.__find_faces(_img_rgb)
        if face_bbs is None:
            return [], []
        return self.__calculate_embedding_from_face(face_bbs, img_rgb)

    def __extend_rectangle(self, rectangle: List[float], frame: np.ndarray, extend_x: float = 0.1, extend_y: float = 0.1)\
            -> Tuple:
        ax, ay, bx, by = rectangle

        origscale_x = bx - ax
        origscale_y = by - ay

        if extend_x > 0:
            ax = max(0, int(ax - extend_x * origscale_x))
            bx = min(frame.shape[1], int(bx + extend_x * origscale_x))

        if extend_y > 0:
            ay = max(0, int(ay - extend_y * origscale_y))
            by = min(frame.shape[0], int(by + extend_y * origscale_y))

        return ax, ay, bx, by

    def get_image_embeddings(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[List[float]]]:
        return self.__calculate_embeddings(image)
