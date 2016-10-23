import platform
from GraphML.configs.graph_config import GraphConfig, GraphSimConfig
from GraphML.utils.label_encoder import BinaryLabelEncoder

graph_max_edge_number = 10000000
label_encoder = BinaryLabelEncoder()
save_progress_interval = 200000
if platform.system() == 'Windows':
	pass
else:
	pass