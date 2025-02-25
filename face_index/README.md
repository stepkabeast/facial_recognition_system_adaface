# face_index
Сервис для индексации вектров и поиска похожих лиц.

- insert_embeddings_to_db
- find_similar_images

## Поднятие сервиса (порт 4445): 
```
docker-compose up --build
```

## Список ENV

    FACE_INDEX_PORT - порт, на котором будет пониматься сервис

Postgres:

    DB_URL - адрес соединения к БД, face_index - схема, которую нужно создать
