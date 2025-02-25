# face_recognizer
Сервис для обнаружения и извлечения вектора признаков из лиц. 

Точки входа:
- Индексатор: /insert_embeddings_to_db
- Поиск по изображениям: /find_similar_documents

```
docker-compose up --build
```

Для запуска сервиса и тестов:
```
test="true" docker-compose up --build
```
