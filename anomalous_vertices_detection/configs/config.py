import os

from anomalous_vertices_detection.utils.label_encoder import BinaryLabelEncoder

label_encoder = BinaryLabelEncoder()
graph_max_edge_number = 10000000
save_progress_interval = 200000

ACADEMIA_URL = "http://proj.ise.bgu.ac.il/sns/datasets/academia.csv.gz"

TWITTER_URL = "http://proj.ise.bgu.ac.il/sns/datasets/twitter.csv.gz"
TWITTER_LABELS_URL = "http://proj.ise.bgu.ac.il/sns/datasets/twitter_fake_ids.csv"

KIDS_URL = "http://proj.ise.bgu.ac.il/sns/datasets/Relationship_patterns_in_the_19th_century.csv"
KIDS_LABELS_URL = "http://proj.ise.bgu.ac.il/sns/datasets/Relationship_patterns_labels.csv"

DATA_DIR_NAME = ".avd"

DATA_DIR = os.path.expanduser(os.path.join('~', DATA_DIR_NAME))
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

TEMP_DIR = os.path.expanduser(os.path.join(DATA_DIR, 'temp'))
if not os.path.exists(TEMP_DIR):
    os.mkdir(TEMP_DIR)
# if not os.access(DATA_DIR, os.W_OK):
#     DATA_DIR = os.path.join(DATA_DIR, '/tmp')
