import sys
import pandas as pandas
import numpy as numpy

arguments = sys.argv[1:]
train_dataset=arguments[1]
test_dataset=arguments[3]
min_leaf_size=arguments[5]
error_type=arguments[6][2:]

train_data=pandas.read_csv(train_dataset)
test_data=pandas.read_csv(test_dataset)


