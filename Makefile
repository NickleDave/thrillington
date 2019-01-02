## all			: make all targets
.PHONY: all
all : data_and_checkpoints

## data_and_checkpoint	: get data, run quick_run_config.ini on it to get checkpoints
.PHONY: data_and_checkpoints
data_and_checkpoints :
	python ./tests/test_data/remake_data_and_checkpoints.py

## clean			: remove auto-generated files
.PHONY: clean
clean :
	rm -rf ./tests/test_data/RAM_results*
	rm -rf ./tests/test_data/mnist
	rm -rf ./tests/test_data/tmp_config.ini

## help			: show help for this Makefile
.PHONY : help
help : Makefile
	@sed -n 's/^##//p' $<
