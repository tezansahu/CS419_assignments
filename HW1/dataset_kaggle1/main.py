import sys
import pandas as pd
import numpy as np
import csv
import math


# Split the tree into two given the column and value to be used for splitting
def split_data(column, value, data):
    print("Calling split_data")
    columns=data.columns.values
    left, right = pd.DataFrame(columns=columns),pd.DataFrame(columns=columns)
    left=data.loc[data[column]<=value]
    right=data.loc[data[column]>value]
    del data
    return left, right


# Calculate the Gini index for a split dataset
# def gini_index(groups, classes):
#     # count all samples at split point
#     n_instances = float(sum([len(group) for group in groups]))
#     # sum weighted Gini index for each group
#     gini = 0.0
#     for group in groups:
#         size = float(len(group))
#         # avoid divide by zero
#         if size == 0:
#             continue
#         score = 0.0
#         # score the group based on the score for each class
#         for class_val in classes:
#             p = [row[-1] for row in group].count(class_val) / size
#             score += p * p
#         # weight the group score by its relative size
#         gini += (1.0 - score) * (size / n_instances)
#     return gini


# Calculate the Cost function for a split dataset
def mean_square_error(lc, rc):
	print("Calling mean_sqr_error")
	mse=0;
	n_tot=len(lc)+len(rc)
	group_sum_sqr=0

	mean_lc=lc.mean(axis=0)[8];
	for row in lc.iterrows():
		group_sum_sqr+=(row[1]["output"]-mean_lc)**2;
	sse+=len(lc)*group_sum_sqr;

	group_sum_sqr=0
	mean_rc=rc.mean(axis=0)[8];
	for row in rc.iterrows():
		group_sum_sqr+=(row[1]["output"]-mean_rc)**2;
	mse+=len(rc)*group_sum_sqr;

	mse/=n_tot
	return mse

def absolute_error(lc, rc):
	print("Calling absolute_error")
	ae=0;
	
	mean=lc.mean(axis=0)[8];
	for row in lc.iterrows():
		ae+=abs(row[1]["output"]-mean)

	mean=rc.mean(axis=0)[8];
	for row in rc.iterrows():
		ae+=abs(row[1]["output"]-mean)
	return ae


# Select the best split value for the dataset returns the node ( containing the split value column, and the two groups)
def best_split(data, loss_function):
    print("Calling best_split")
    best_column, best_value, best_cost, best_groups = None, float('inf'), float("inf"), None
    for column in data.columns.values[:-1]:
        for row in data.iterrows():
            left_child, right_child = split_data(column, row[1][column], data)  # calls split data properly
            if(loss_function=="absolute"):
                cost = absolute_error(left_child, right_child)
            elif(loss_function=="mean_squared"):
                cost = mean_square_error(left_child, right_child)

        if cost < best_cost:
            best_column, best_value, best_cost, best_lc, best_rc = column, row[1][column], cost, left_child, right_child
    print(best_column,best_value, best_lc.head(), best_rc.head())
    return {'column': best_column, 'value': best_value, 'left_child': best_lc, 'right_child': best_rc}



# Create a terminal node value
def create_leaf(group):
	print("Calling create_leaf")
	#nodeval = sum([g[-1] for g in group]) / len(group)
    #return nodeval
	mean=group.mean(axis=0)[8];
	return {'column': None, 'value': None, 'left_child': None, 'right_child': None, 'prediction':mean}


# Spliting the tree  recursively at a particular node or fixing a terminal
def recursive_split(node, max_depth, min_records, cur_depth, loss_func):
    print("Calling recursive_split")
    left, right = node['left_child'], node['right_child']


    # if the best split is a no split
    if len(left)==0 or len(right)==0:
        print("No split")
        node['left_child'] = node['right_child'] = create_leaf(left.append(right))
        return

    # check for max depth
    if cur_depth >= max_depth:
        print("Max depth reached")
        node['left_child'], node['right_child'] = create_leaf(left), create_leaf(right)
        return

    # split left side if greater than minimum records
    if len(left) <= min_records:
        print("left_child_min_nodes_reached")
        node['left_child'] = create_leaf(left)
    else:
        node['left_child'] = best_split(left, loss_func)
        recursive_split(node['left_child'], max_depth, min_records, cur_depth + 1, loss_func)

    # split right side if greater than minimum records
    if len(right) <= min_records:
        print("right_child_min_nodes_reached")
        node['right_child'] = create_leaf(right)
    else:
        node['right_child'] = best_split(right, loss_func)
        recursive_split(node['right_child'], max_depth, min_records, cur_depth + 1, loss_func)


# Build a decision tree
def build_tree(dataset, max_depth, min_records, loss_func):
	print("Calling build_tree")
	root_node = best_split(dataset, loss_func)
	recursive_split(root_node, max_depth, min_records, 1, loss_func)
	return root_node


# Predict the output of a data value using the decision tree
def predict_output(node, test_record):
	print("Calling predict_output")
	if test_record[node['column']] < node['value']:  # checks if belongs to node's left or right 
		if isinstance(node['left'], dict):  # check if its dictionary or not, dictionary means it has subnodes
			predict_output(node['left'], test_record)
		else:
			return node['left']  # its a terminal node return it
	else:
		if isinstance(node['right'], dict):
			predict_output(node['right'], test_record)
		else:
			return node['right']


# Print the tree structure
def print_tree(node, depth=0):
	if node['left_child']!=None and node['right_child']!=None:
		print('%s[X%s < %.3f]' % (depth * ' ', (node['column']), node['value']))
		print_tree(node['left_child'], depth + 1)
		print_tree(node['right_child'], depth + 1)
	else:
		print('%s[%s]' % ((depth * ' ', node)))


'''Main Program'''

arguments = sys.argv[1:]
train_dataset=arguments[1]
test_dataset=arguments[3]
min_leaf_size=int(arguments[5])
loss_function=arguments[6][2:]

print("Recieved arguments")

train_data=pd.read_csv(train_dataset, header=0)
test_data=pd.read_csv(test_dataset, header=0)
print("Read all data")

rootnode = build_tree(train_data, 1, min_leaf_size, loss_function)
print_tree(rootnode)


'''
For a continuous variable :
change the cost function
change the final node creation to take average of the output values

'''

