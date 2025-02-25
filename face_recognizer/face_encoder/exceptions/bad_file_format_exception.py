from typing import Optional

from face_encoder.exceptions.base_exception import ServiceBaseException


class BadFileFormatException(ServiceBaseException):
    """
    Raise if given file can't be handled by the system (for example if no reader can read this file)
    """

    def __init__(self, msg: str, filename: Optional[str] = None) -> None:
        super(BadFileFormatException, self).__init__(msg=msg, filename=filename)

    def __str__(self) -> str:
        return "BadFileException({})".format(self.msg)

    @property
    def code(self) -> int:
        return 415
