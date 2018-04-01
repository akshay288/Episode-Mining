# Episode Mining

## Setup
```bash
$ make env
$ source env/bin/activate
$ make install
```

## Usage
To run a disaggregator, run `python3 scripts/run.py`.

To optimize a certain algorithm, run `python3 scripts/optimize.py`. The different datasources and disaggregators can be changed in the script.

# Development
To create a new datasource, follow the spec `episode_mining/datasources/base.py` provides. Any of the datasources in `episode_mining/datasources` can be used as examples. 

A new disaggregator can be implemented by following the spec in `episode_mining/disaggregators/base.py`.