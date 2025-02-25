from typing import Optional


class ServiceBaseException(Exception):
    def __init__(self, msg: str, filename: Optional[str] = None) -> None:
        super(ServiceBaseException, self).__init__()
        self.msg = msg
        self.filename = filename

    def __str__(self) -> str:
        return "BaseException({})".format(self.msg)

    @property
    def code(self) -> int:
        return 400
