The Anomalous-Vertices-Detection project is a Python package for performing graph analysis.
The package supports extracting graphs'  topological features, performing link prediction, and identifying anomalous vertices.
The package supports various graph packages (NetworkX, SGraph, iGraph, and GraphTools) and Machine Learning packages (SciKit and GraphLab).
This project is under development and has a many planned improvements. More details on the project can be find in the our paper titled "Unsupervised Anomalous Vertices Detection UtilizingLink Prediction Algorithms".

##Installation
```
git clone git://github.com/Kagandi/anomalous-vertices-detection.git
pip install -r requirements.txt
```
GraphLab may require installation of additional requirements, if needed a message with instruction will be presented.
##Usage
Init:
```python
labels = {"neg": "Real", "pos": "Fake"}
cls = GlLearner(labels=labels) #GraphLab Learner
dataset_config = GraphConfig("my_dataset", my_dataset, is_directed=True)
gl = GraphLearningController(GlLearner(labels=labels), labels, dataset_config)
my_graph = GraphFactory().make_graph_with_fake_profiles(dataset_config.data_path,
                                            is_directed=dataset_config.is_directed,
                                            pos_label=labels["pos"], neg_label=labels["neg"])

```

##Todo
- [ ] Complete documentation
- [ ] Write Jupiter notebooks
- [ ] Clean the code
- [ ] Add setup.py
- [X] Add requirements.txt
- [ ] Add examples
- [ ] Add more test
- [ ] Python 3.5 support


