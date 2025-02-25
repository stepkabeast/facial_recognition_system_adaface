import os
from collections import namedtuple
from typing import Tuple, List

import numpy as np
from logging import getLogger

from face_encoder.external_services.s3_client import S3Client
from tests.abstract_api_test import AbstractTestApi

Person = namedtuple("Person", ["image_path", "uuid"])


class TestAPIFullLfwDataset(AbstractTestApi):

    client = S3Client(getLogger())
    bucket_name = "talismandocumentbucket"

    def __calculate_accuracy(self, predict_issame: np.ndarray, actual_issame: np.ndarray):
        tp = np.sum(np.logical_and(predict_issame, actual_issame))
        fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
        tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
        fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
        tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
        fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
        frr = 0 if (tp + fn == 0) else float(fn) / float(tp + fn)
        acc = float(tp + tn) / predict_issame.size

        return tpr, fpr, frr, acc

    def __parse_lfw_folder(self, path: str) -> Tuple[Person, List[Person]]:
        etalon_person = None
        positive_persons = []
        file_names = os.listdir(path)

        for name in file_names:
            if not os.path.isfile(os.path.join(path, name)):
                continue

            if name.endswith("_0001.jpg"):
                etalon_person = Person(image_path=os.path.join(path, name), uuid=name)
            else:
                positive_persons.append(Person(image_path=os.path.join(path, name), uuid=name))
        return etalon_person, positive_persons

    def __insert_person_to_db(self, person: Person):
        file_name = person.image_path
        uuid = person.uuid
        self.client.check_bucket(self.bucket_name)
        path = self._get_abs_path(file_name)

        with open(path, "rb") as f:
            self.client.upload(self.bucket_name, file_name, f)

        tdm = self._make_tdm(f"s3://{self.bucket_name}/{file_name}", uuid)
        config = {}

        self._insert_embeddings(tdm, config)

    def __test_find_similar_documents_lfw(self, persons: List[Person]) -> np.ndarray:
        ids = []
        predicted_ids = []
        self.client.check_bucket(self.bucket_name)

        for person in persons:
            uuid = person.uuid
            file_name = person.image_path
            path = self._get_abs_path(file_name)

            with open(path, "rb") as f:
                self.client.upload(self.bucket_name, file_name, f)

            tdm = self._make_tdm(f"s3://{self.bucket_name}/{file_name}", uuid)
            result = self._find_similar_images(tdm, config={})

            if result.get("metadata") is None:
                continue

            similar_documents_ids = result.get("metadata").get("similar_documents_ids")

            if similar_documents_ids is None:
                continue

            for similar_uuid in similar_documents_ids:
                size = len(uuid)
                uuid = uuid[:size - 9]
                ids.append(uuid)
                size = len(similar_uuid)
                similar_uuid = similar_uuid[:size - 9]
                predicted_ids.append(similar_uuid)

        predicted_ids = np.array(predicted_ids)
        predicted_issame = np.equal(ids, predicted_ids)
        return predicted_issame

    def test_lfw(self):
        data_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "tests/data"))
        data_lfw = os.path.join(data_dir, 'lfw_full')
        etalon_persons = []
        positive_persons = []
        for (dir_path, dir_names, file_names) in os.walk(data_lfw):
            for idx, person_dir in enumerate(dir_names):
                etalon, positive = self.__parse_lfw_folder(os.path.join(dir_path, person_dir))
                self.__insert_person_to_db(etalon)
                etalon_persons.append(etalon)
                positive_persons.extend(positive)

        predict_issame = self.__test_find_similar_documents_lfw(positive_persons)

        actual_issame = np.empty(len(predict_issame))
        actual_issame.fill(True)

        tpr, fpr, frr, acc = self.__calculate_accuracy(predict_issame, actual_issame)
        print("RESULT EVALUATION:\n TPR={}\n FPR={}\n FRR={}\nAccuracy={}".format(tpr, fpr, frr, acc))
