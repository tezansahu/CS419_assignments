import sys
import pandas as pd
import numpy as np
import csv



# Split the tree into two given the column and value to be used for splitting
def split_data(column_num, value, data):
    left, right = list(), list()
    for record in data:
        if record[column_num] < value:
            left.append(record)
        else:
            right.append(record)
    return left, right


# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini


# Calculate the Cost function for a split dataset
def sum_square_error(groups):
    sse=0;
    for group in groups:
    	mean=group.mean(axis=0)[8];
    	for row in group.iterrows():
    		sse+=(row[1]["output"]-mean)^2;

    return sse

 def absolute_error(groups):
 	ae=0;
    for group in groups:
    	mean=group.mean(axis=0)[8];
    	for row in group.iterrows():
    		ae+=abs(row[1]["output"]-mean)

 	return ae


# Select the best split value for the dataset returns the node ( containing the split value column, and the two groups)
def best_split(data):
    output_values = list(set(d[-1] for d in data))
    best_column, best_value, best_cost, best_groups = 999, 999, 999, None
    for column_num in range(len(data[0]) : - 1):
        for record in data:
            groups = split_data(column_num, record[column_num], data)  # calls split data properly
            cost = gini_index(groups, output_values)
            '''
            change the function cost_func
            '''
            # print('X%d < %.3f  Cost=%.3f' %( (column_num+1), record[column_num], cost))
            if cost < best_cost:
                best_column, best_value, best_cost, best_groups = column_num, record[column_num], cost, groups
    return {'column': best_column, 'value': best_value, 'groups': best_groups}


# Create a terminal node value
def createnode(group):
    #nodeval = sum([g[-1] for g in group]) / len(group)
    #return nodeval
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


# Spliting the tree  recursively at a particular node or fixing a terminal
def recursive_split(node, max_depth, min_records, cur_depth):
    left, right = node['groups']
    del (node['groups'])

    # if the best split is a no split
    if not left or not right:
        node['left'] = node['right'] = createnode(left + right)
        return

    # check for max depth
    if cur_depth >= max_depth:
        node['left'], node['right'] = createnode(left), createnode(right)
        return

    # split left side if greater than minimum records
    if len(left) <= min_records:
        node['left'] = createnode(left)
    else:
        node['left'] = best_split(left)
        recursive_split(node['left'], max_depth, min_records, cur_depth + 1)

    # split right side if greater than minimum records
    if len(right) <= min_records:
        node['right'] = createnode(right)
    else:
        node['right'] = best_split(right)
        recursive_split(node['right'], max_depth, min_records, cur_depth + 1)


# Build a decision tree
def build_tree(dataset, max_depth, min_records):
    root_node = best_split(dataset)
    recursive_split(root_node, max_depth, min_records, 1)
    return root_node


# Predict the output of a data value using the decision tree
def predict_output(node, test_record):
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
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % (depth * ' ', (node['column'] + 1), node['value']))
        print_tree(node['left'], depth + 1)
        print_tree(node['right'], depth + 1)
    else:
        print('%s[%s]' % ((depth * ' ', node)))


'''Main Program'''

arguments = sys.argv[1:]
train_dataset=arguments[1]
test_dataset=arguments[3]
min_leaf_size=arguments[5]
error_type=arguments[6][2:]

train_data=pd.read_csv(train_dataset, header=0)
test_data=pd.read_csv(test_dataset, header=0)

rootnode = build_tree(train_data, 4, 1)
print_tree(rootnode)


'''
For a continuous variable :
change the cost function
change the final node creation to take average of the output values

'''

