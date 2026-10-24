.PHONY: install data train-all train-rl train-lstm train-gan evaluate evaluate-rl evaluate-lstm app test layout-experiment scenario

PYTHON ?= python

install:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) scripts/prepare_data.py

train-all:
	$(PYTHON) scripts/train_lstm.py
	$(PYTHON) scripts/train_rl.py
	$(PYTHON) scripts/train_gan.py


train-rl:
	$(PYTHON) scripts/train_rl.py

train-lstm:
	$(PYTHON) scripts/train_lstm.py

train-gan:
	$(PYTHON) scripts/train_gan.py

evaluate:
	$(PYTHON) scripts/evaluate_all.py

evaluate-rl:
	$(PYTHON) scripts/evaluate_rl.py

evaluate-lstm:
	$(PYTHON) scripts/evaluate_lstm.py

app:
	$(PYTHON) -m app.main

test:
	$(PYTHON) -m pytest

layout-experiment:
	$(PYTHON) scripts/run_layout_experiment.py

scenario:
	$(PYTHON) scripts/run_scenario.py
