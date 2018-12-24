.PHONY: all clean

clean :
	rm -rf ./tests/test_data/checkpoints
	rm -rf ./tests/test_data/mnist

all : data_and_checkpoints

data_and_checkpoints :
	python ./tests/test_data/remake_data_and_checkpoints.py
