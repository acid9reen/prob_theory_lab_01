default: main_window.py hypothesis_dialogue.py
auto_gen_code_folder := ../src/widgets_autogen/

main_window.py: main_window.ui
	pyuic5 $^ -o $@
	mv $@ $(auto_gen_code_folder)

hypothesis_dialogue.py: hypothesis_dialogue.ui
	pyuic5 $^ -o $@
	mv $@ $(auto_gen_code_folder)
