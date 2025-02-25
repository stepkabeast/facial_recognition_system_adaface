import os
import psycopg2

from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.testing.provision import drop_db
from sqlalchemy_utils import database_exists, create_database
from logging.handlers import RotatingFileHandler

from pg_model.meta import Base, PortfolioEmbeddingInfo


def tables_init_db(eng, connection):

    if not database_exists(eng.url):
        create_database(eng.url)

    db_ses = scoped_session(sessionmaker(autocommit=False,
                                         autoflush=False,
                                         bind=eng))

    Base.query = db_ses.query_property()

    connection = eng.connect()
    '''trans = conn.begin()
    for table_name in Base.metadata.tables.keys():
        conn.execute("DROP TABLE IF EXISTS {0} CASCADE".format(table_name))
    conn.execute("DROP TABLE IF EXISTS apscheduler_jobs CASCADE;")

    for table in Base.metadata.tables.values():
        for column in table.c:
            if isinstance(column.type, Enum):
                conn.execute("DROP TYPE IF EXISTS {0}".format(column.type.name))
    trans.commit()'''
    Base.metadata.drop_all(bind=eng)
    db_ses.commit()

    connection.execute("drop extension if exists cube cascade;")
    connection.execute("create extension cube;")
    Base.metadata.create_all(bind=eng)
    db_ses.query(PortfolioEmbeddingInfo).delete()
    db_ses.commit()

    rows = db_ses.query(PortfolioEmbeddingInfo).all()
    print("Now {} rows in DocumentEmbeddingInfo table".format(len(rows)))

    return db_ses


if __name__ == "__main__":
    is_init = os.environ.get("IS_INIT", "True")
    hdlr = RotatingFileHandler("db_init.log", mode="a", encoding="utf-8", maxBytes=20000000, backupCount=10, delay=True)

    if is_init == "True":
        connection_url = os.environ.get("DB_URL", "postgresql://postgres:postgres@127.0.0.1:5432/face_index")
        engine = create_engine(connection_url, encoding='utf-8')

        if not database_exists(engine.url):
            create_database(engine.url)

        DB_CONFIG = {"database": connection_url.split("/")[-1],
                     "user": connection_url.split(":")[1][2:],
                     "password": connection_url.split(":")[2].split("@")[0],
                     "host": connection_url.split("@")[1].split(":")[0],
                     "port": connection_url.split("@")[1].split(":")[1].split("/")[0],
                     }

        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        db_session = tables_init_db(engine, conn)
        db_session.close()
        cur.close()
        conn.close()
