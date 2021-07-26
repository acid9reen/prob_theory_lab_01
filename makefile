ui_dir := ./prob_theory_lab_01/ui
venv_dir := ./.venv
requirements := ./requirements.txt
python := python3

default: help

help:
	@echo "This is the help menu:"
	@echo "	make ui - to update ui files"
	@echo "	make venv - to create venv"
	@echo "	make requirements - to install/update requirements"
	@echo "	make setup - to create venv and install requirements"

setup: venv requirements

venv:
	$(python) -m venv $(venv_dir)

requirements: $(requirements)
	. $(venv_dir)/bin/activate
	pip install -r $^

ui: $(ui_dir)
	$(MAKE) --directory=$^

clean:
	rm -rf $(venv_dir)
	find -iname "*.pyc" -delete
