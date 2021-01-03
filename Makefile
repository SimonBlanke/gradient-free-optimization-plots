dist:
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

install:
	pip install .

develop:
	pip install -e .

reinstall:
	pip uninstall -y optimizationplotlib
	rm -fr build dist optimizationplotlib.egg-info
	python setup.py bdist_wheel
	pip install dist/*
