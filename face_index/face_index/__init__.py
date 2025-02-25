from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy_utils import database_exists, create_database
import os
from pg_model.meta import Base


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class DbSession(metaclass=Singleton):

    def __init__(self):
        connection_url = os.environ.get("DB_URL", "postgresql://postgres:postgres@127.0.0.1:5432/face_index")
        self.engine = create_engine(connection_url, pool_size=50)

        if not database_exists(self.engine.url):
            create_database(self.engine.url)

        if not self.engine.dialect.has_table(self.engine.connect(), "face_embedding_info"):
            Base.metadata.create_all(bind=self.engine)

    def get_new_session(self):
        db_session = scoped_session(sessionmaker(autocommit=False,
                                                 autoflush=False,
                                                 bind=self.engine))

        Base.query = db_session.query_property()

        return db_session


db_session_creator = DbSession()
