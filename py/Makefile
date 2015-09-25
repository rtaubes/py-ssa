.PHONY: all run clean

all : ui_main_dlg.py ui_coeff_dlg.py run

ui_main_dlg.py : main_dlg.ui
	pyuic4 -o $@ $<

ui_coeff_dlg.py : coeff.ui
	pyuic4 -o $@ $<

run:
	./ssa.py

clean:
	@find . \( -name "*.pyc" -o -name ".*.swp" -o -name ".*.swo" -o -name ".*~" -o -name "*~" \) -exec rm -fv '{}' ';'
