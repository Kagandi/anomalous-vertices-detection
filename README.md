anomalous-vertices-detection is a generic graph machine learning project.
The library support many different graph packages(NetworkX, GraphLab, GraphTools)  and ML packages (SciKit, GraphLab).
This project is still under development and has a many palnned improvments.

##Download
```
$ git clone git://github.com/Kagandi/anomalous-vertices-detection.git
```
##Usage
Init:
```python
labels = {"neg": "Real", "pos": "Fake"}
cls = GlLearner(labels=labels) #GraphLab Learner
dataset_config = GraphConfig("my_dataset", my_dataset, is_directed=True)
my_graph = gf.make_graph(dataset_config.data_path, is_directed=dataset_config.is_directed, pos_label=labels["pos"],
						 neg_label=labels["neg"], start_line=dataset_config._first_line,
						 delimiter=dataset_config.delimiter, package="Networkx")
```

##Todo
- [ ] Complete documentation
- [ ] Clean the code
- [ ] Add setup.py
- [ ] Add requirements.txt
- [ ] Add examples
- [ ] Add more test


