from typing import Optional

from face_encoder.exceptions.base_exception import ServiceBaseException


class S3Exception(ServiceBaseException):
    """
    raise if there is no file in s3-storage
    """

    def __init__(self, msg: str, filename: Optional[str] = None) -> None:
        super(S3Exception, self).__init__(msg=msg, filename=filename)

    def __str__(self) -> str:
        return f"S3Exception({self.msg})"

    @property
    def code(self) -> int:
        return 404
