venv: reqs
	virtualenv venv && . venv/bin/activate && pip install -r requirements.txt

reqs:
	pip-compile reqs.in -o requirements.txt -v

make_venv:
	if [ ! -d "venv" ]; then \
		virtualenv venv; \
	else \
		echo "Virtual environment already exists."; \
	fi

install_deps: 
	. venv/bin/activate && pip install -r requirements.txt

setup: reqs make_venv install_deps