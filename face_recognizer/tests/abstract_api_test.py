import json
import logging
import os
import unittest
import requests
from starlette.responses import JSONResponse
from face_encoder.manager import get_face_index_port, get_face_index_host


class AbstractTestApi(unittest.TestCase):
    data_directory_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
    logger = logging.getLogger()
    os.environ.setdefault("IS_TEST", "true")

    def _get_host(self) -> str:
        host = os.environ.get('FACE_RECOGNIZER_HOST', 'localhost')
        return host

    def _get_port(self) -> int:
        port = int(os.environ.get('IMAGE_RECOGNIZER_PORT', '4446'))
        return port

    def _get_abs_path(self, file_name: str) -> str:
        return os.path.join(self.data_directory_path, file_name)


    def _send_request(self, file_name: str, data: dict = None, expected_code: int = 200) -> dict:
        """
        send file `file_name` in post request with `data` as parameters. Expects that response return code
        `expected_code`

        :param file_name: name of file (should lie  src/tests/data folder
        :param data: parameter dictionary (here you can put language for example)
        :param expected_code: expected http response code. 200 for normal request
        :return: result from json
        """
        if data is None:
            data = {}

        host = self._get_host()
        port = self._get_port()
        abs_path = self._get_abs_path(file_name)

        with open(abs_path, 'rb') as file:
            files = {'file': (file_name, file)}
            r = requests.post(f"http://{host}:{port}/encode-faces", files=files, data=data)
            self.assertEqual(expected_code, r.status_code)
            return json.loads(r.content.decode())

    def _insert_embeddings(self, uuid: str, s3_path: str, config: dict, expected_code: int = 200) -> JSONResponse:
        host = self._get_host()
        port = self._get_port()
        data = {
            "message": "",
            "uuid": uuid,
            "config": config,
            "s3_path": s3_path

        }

        r = requests.post(f"http://{host}:{port}/insert_embeddings_to_db", json=data)
        self.assertEqual(expected_code, r.status_code)

        return JSONResponse({"result": True})

    def _find_similar_images(self, s3_path: str, config: dict, expected_code: int = 200) -> dict:
        host = self._get_host()
        port = self._get_port()
        data = {
            "message": "",
            "config": config,
            "s3_path": s3_path
        }

        r = requests.post(f"http://{host}:{port}/find_similar_images", json=data)
        self.assertEqual(expected_code, r.status_code)

        return json.loads(r.content.decode())

    def _call_index_build(self, config: dict, expected_code: int = 200) -> dict:
        host = get_face_index_host()
        port = get_face_index_port()
        data = {
            "message": "",
            "config": config
        }

        r = requests.post(f"http://{host}:{port}/init_index", json=data)
        self.assertEqual(expected_code, r.status_code)

        return json.loads(r.content.decode())
