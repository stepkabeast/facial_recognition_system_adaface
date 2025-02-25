import logging
import os
import sys
from logstash_async.handler import AsynchronousLogstashHandler

DEBUG_MODE = False

logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format="%(asctime)s - %(pathname)s - %(levelname)s - %(message)s")

logger = logging.getLogger('python-logstash-logger')
logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)

if "log_host" in os.environ:
    host = os.environ["log_host"]
    port = int(os.environ["log_port"])
    logger.addHandler(AsynchronousLogstashHandler(host=host, port=port, database_path=None))
