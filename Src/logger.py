import logging 
import os 
from datetime import datetime


LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOGS_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOGS_DIR, LOG_FILE)


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(lineno)d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename=LOG_FILE_PATH
)

# Function to create named loggers
def get_logger(name: str):
    return logging.getLogger(name)

if __name__ == "__main__":
    test_logger = get_logger(__name__)
    test_logger.info("Logger setup complete")
    
