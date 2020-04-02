dist:
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

install:
	pip install .

develop:
	pip install -e .

reinstall:
	pip uninstall -y hyperplotlib
	rm -fr build dist hyperplotlib.egg-info
	python setup.py bdist_wheel
	pip install dist/*