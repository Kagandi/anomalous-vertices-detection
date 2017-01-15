The Anomalous-Vertices-Detection project is a Python package for performing graph analysis.
The package supports extracting graphs'  topological features, performing link prediction, and identifying anomalous vertices.
The package supports various graph packages ([NetworkX](https://networkx.github.io), [SGraph](https://turi.com/products/create/docs/generated/graphlab.SGraph.html), [iGraph](http://igraph.org/python/), and
[GraphTools](https://graph-tool.skewed.de/)) and Machine Learning packages ([SciKit](http://scikit-learn.org/) and [GraphLab](https://turi.com/products/create/docs/index.html)).
This project is under development and has a many planned improvements. More details on the project can be find in the our paper titled "Unsupervised Anomalous Vertices Detection Utilizing Link Prediction Algorithms".

##Installation
```
git clone git://github.com/Kagandi/anomalous-vertices-detection.git
pip install -r requirements.txt
python setup.py install
```
GraphLab may require installation of additional requirements, if needed a message with instruction will be presented.
##Usage
Init:
```python
from anomalous_vertices_detection.configs.graph_config import GraphConfig
from anomalous_vertices_detection.graph_learning_controller import *
from anomalous_vertices_detection.graphs.graph_factory import GraphFactory
from anomalous_vertices_detection.learners.gllearner import GlLearner

labels = {"neg": "Real", "pos": "Fake"}
dataset_config = GraphConfig("my_dataset", my_dataset, is_directed=True)
gl = GraphLearningController(GlLearner(labels=labels), labels, dataset_config)
my_graph = GraphFactory().make_graph_with_fake_profiles(dataset_config.data_path,
                                            is_directed=dataset_config.is_directed,
                                            pos_label=labels["pos"], neg_label=labels["neg"])

```

##Todo
- [ ] Complete the documentation
- [ ] Write Jupiter notebooks
- [ ] Clean the code
- [X] Add setup.py
- [X] Add requirements.txt
- [X] Add basic examples
- [ ] Add more examples
- [ ] Add more test
- [ ] Python 3.5 support


