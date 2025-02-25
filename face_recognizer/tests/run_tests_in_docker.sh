echo "GO"
if [ "$IS_TEST" = "true" ]
  then
    echo "run tests"
    response=$(curl http://$FACE_RECOGNIZER_HOST:$FACE_RECOGNIZER_PORT/ --silent)
    echo http://$FACE_RECOGNIZER_HOST:$FACE_RECOGNIZER_PORT/
    until [ "$response" = "200" ]; do
      response=$(curl --write-out %{http_code} --silent --output /dev/null "http://$FACE_RECOGNIZER_HOST:$FACE_RECOGNIZER_PORT/")
      >&2 echo "FACE RECOGNIZER is unavailable - sleeping, response: $response"
      sleep 5
    done
    echo "FACE RECOGNIZER is up"
    python3.8 -m unittest -v -f /face_recognizer_root/tests/test*

else
  echo "skip tests"
  echo "if you want to run tests do "
  echo 'test="true" docker-compose up  --build --exit-code-from test'
  sleep 10
fi