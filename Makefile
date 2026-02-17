.PHONY: venv install install-train

venv:
	python3 -m venv .venv

install:
	. .venv/bin/activate && pip install --upgrade pip && pip install -e .

install-train:
	. .venv/bin/activate && pip install --upgrade pip && pip install -e .[train]
