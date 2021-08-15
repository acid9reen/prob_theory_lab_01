ui_dir := ./ui
venv_dir := ./.venv
requirements := ./requirements/dev.txt
python := python3

default: help

help:
	@echo "This is the help menu:"
	@echo "	make gui - to update ui files"
	@echo "	make venv - to create venv"
	@echo "	make requirements - to install/update requirements"
	@echo "	make setup - to create venv and install requirements"

setup: venv requirements

venv:
	$(python) -m venv $(venv_dir)

requirements: $(requirements)
	. $(venv_dir)/bin/activate
	pip install -r $^

gui: $(ui_dir)
	$(MAKE) --directory=$^

clean:
	rm -rf $(venv_dir)
	find -iname "*.pyc" -delete
