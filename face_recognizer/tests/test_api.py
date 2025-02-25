import json
import numpy as np
from logging import getLogger

from face_encoder.external_services.s3_client import S3Client
from tests.abstract_api_test import AbstractTestApi


class TestApiFaceRecognizer(AbstractTestApi):
    client = S3Client(getLogger())
    bucket_name = "talismandocumentbucket"

    def setUp(self) -> None:
        super().setUp()
        files = {
            "1/0.jpeg",
            "2/0.jpeg",
            "3/0.jpeg",
            "4/0.jpeg",
            "5/0.jpeg",
            "6/0.jpeg",
            "7/0.jpeg",
            "8/0.jpeg",
            "9/0.jpeg",
            "10/0.jpeg",
            "6/1495276406.jpg"
        }
        self.client.check_bucket(self.bucket_name)
        for file_name in files:
            path = self._get_abs_path(file_name)
            with open(path, "rb") as f:
                self.client.upload(self.bucket_name, file_name, f)

            s3_path = f"s3://{self.bucket_name}/{file_name}"
            self._insert_embeddings(file_name, s3_path, config={})

    def tearDown(self) -> None:
        super().tearDown()
        self.client.remove_bucket(bucket_name=self.bucket_name)

    def test_topn(self) -> None:
        file_name = "11/1407075512.jpg"
        path = self._get_abs_path(file_name)
        with open(path, "rb") as f:
            self.client.upload(self.bucket_name, file_name, f)

        s3_path = f"s3://{self.bucket_name}/{file_name}"
        result = self._find_similar_images(s3_path, config={"topn": 100000, "threshold": 2})

        self.assertEqual("0", '4/0.jpeg')

