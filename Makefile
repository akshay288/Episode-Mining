env: 
	virtualenv -p $$(which python3) env

install:
	g++ genetic_k_means/GeneticKMeansAlgorithm.cpp -o genetic_k_means/gka
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

clean-pyc:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf