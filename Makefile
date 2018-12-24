.PHONY: all clean

clean :
	rm -rf ./tests/test_data/checkpoints
	rm -rf ./tests/test_data/mnist

all : models

checkpoints : data
	python ./tests/test_data/remake_checkpoints.py

data :
python ./tests/test_data/feature_files/remake_data.py
