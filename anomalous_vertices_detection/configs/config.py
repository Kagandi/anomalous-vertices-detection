import os
from os.path import join, dirname
from dotenv import load_dotenv

from anomalous_vertices_detection.utils.label_encoder import BinaryLabelEncoder

label_encoder = BinaryLabelEncoder()
graph_max_edge_number = 10000000
save_progress_interval = 200000

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

ACADEMIA_URL = os.environ.get("ACADEMIA_URL")

TWITTER_URL = os.environ.get("TWITTER_URL")
TWITTER_LABELS_URL = os.environ.get("TWITTER_LABELS_URL")
DATA_DIR_NAME = os.environ.get("LOCAL_DATA_FOLDER_NAME")
DATA_DIR = os.path.expanduser(os.path.join('~', DATA_DIR_NAME))
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

TEMP_DIR = os.path.expanduser(os.path.join(DATA_DIR, 'temp'))
if not os.path.exists(TEMP_DIR):
    os.mkdir(TEMP_DIR)
# if not os.access(DATA_DIR, os.W_OK):
#     DATA_DIR = os.path.join(DATA_DIR, '/tmp')
