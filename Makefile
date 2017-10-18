run-tests:
  pipenv run python setup.py install
  pipenv run python -m  unittest discover
init:
    pip install --upgrade pip
	python setup.py install
	pipenv install --dev