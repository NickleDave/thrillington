.PHONY: all clean

clean :
	rm -rf ./tests/test_data/checkpoints
	rm -rf ./tests/test_data/mnist
	rm .tests/test_data/tmp_config.ini

all : data_and_checkpoints

data_and_checkpoints :
	python ./tests/test_data/remake_data_and_checkpoints.py
