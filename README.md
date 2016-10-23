anomalous-vertices-detection is a generic graph machine learning project.
The library support many different graph packages(NetworkX, GraphLab, GraphTools)  and ML packages (SciKit, GraphLab).
This project is still under development and has a lot of work ahead.

##Download
```
$ git clone git://github.com/Kagandi/anomalous-vertices-detection.git
```
##Usage
Init:
```python
labels = {"neg": "Real", "pos": "Fake"}
cls = GlLearner(labels=labels)
dataset_config = GraphConfig("my_dataset", my_dataset, is_directed=True)

```
##Limitations
* The codes was fully tested with NxGraph (NetworkX) as graph and GraphLab as machine learning modules.
* GtGraph (graph-tools) was implemented long time ago, I have to revisit the code to see if it works with latest graph-tools version.
* After Turi was brought by apple it is not clear the future of the package. 
* SGraph cannot be in this algorithm since it doesn’t support removal of edges. However it can be used to extract features for other purposes.

##Todo
- [ ] Complete documentation
- [ ] Clean the code
- [ ] Add setup.py
- [ ] Add requirements.txt
- [ ] Add examples
- [ ] Add more test


