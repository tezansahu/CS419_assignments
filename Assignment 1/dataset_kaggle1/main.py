import sys
import pandas as pd



# Split the tree into two given the column and value to be used for splitting
def split_data(column, value, data):
    # # print("Calling split_data")
    columns=data.columns.values
    left, right = pd.DataFrame(columns=columns),pd.DataFrame(columns=columns)
    left=data.loc[data[column]<=value]
    right=data.loc[data[column]>value]
    del data
    return left, right


# Calculate the Cost function for a split dataset
def mean_square_error(lc, rc):
	# # print("Calling mean_sqr_error")
	mse=0;
	n_tot=len(lc)+len(rc)
	group_sum_sqr=0

	mean_lc=lc.mean(axis=0)[8];
	for row in lc.iterrows():
		group_sum_sqr+=(row[1]["output"]-mean_lc)**2;
	mse+=group_sum_sqr;

	group_sum_sqr=0
	mean_rc=rc.mean(axis=0)[8];
	for row in rc.iterrows():
		group_sum_sqr+=(row[1]["output"]-mean_rc)**2;
	mse+=group_sum_sqr;

	
	return mse

def absolute_error(lc, rc):
	# # print("Calling absolute_error")
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
    # # print("Calling best_split")
    cost=float('inf')
    best_column, best_value, best_cost, best_groups = None, float('inf'), float("inf"), None
    best_lc, best_rc=None, None
    for column in data.columns.values[:-1]:
        for row in data.iterrows():
            left_child, right_child = split_data(column, row[1][column], data)  # calls split data properly
            if(loss_function=="absolute"):
                cost = absolute_error(left_child, right_child)
            elif(loss_function=="mean_squared"):
                cost = mean_square_error(left_child, right_child)

        if cost < best_cost:
            best_column, best_value, best_cost, best_lc, best_rc = column, row[1][column], cost, left_child, right_child
    # # print(best_column,best_value, best_lc.head(), best_rc.head())
    return {'column': best_column, 'value': best_value, 'left_child': best_lc, 'right_child': best_rc}



# Create a terminal node value
def create_leaf(group):
	# print("Calling create_leaf")
	#nodeval = sum([g[-1] for g in group]) / len(group)
    #return nodeval
	mean=group.mean(axis=0)[8];
	return {'column': None, 'value': None, 'left_child': None, 'right_child': None, 'prediction':mean}


# Spliting the tree  recursively at a particular node or fixing a terminal
def recursive_split(node, max_depth, min_records, cur_depth, loss_func):
    # print("Calling recursive_split")
    left, right = node['left_child'], node['right_child']


    # if the best split is a no split
    if len(left)==0 or len(right)==0:
        # print("No split")
        node['left_child'] = node['right_child'] = create_leaf(left.append(right))
        return

    # check for max depth
    if cur_depth >= max_depth:
        # print("Max depth reached")
        node['left_child'], node['right_child'] = create_leaf(left), create_leaf(right)
        return

    # split left side if greater than minimum records
    if len(left) <= min_records:
        # print("left_child_min_nodes_reached")
        node['left_child'] = create_leaf(left)
    else:
        node['left_child'] = best_split(left, loss_func)
        recursive_split(node['left_child'], max_depth, min_records, cur_depth + 1, loss_func)

    # split right side if greater than minimum records
    if len(right) <= min_records:
        # print("right_child_min_nodes_reached")
        node['right_child'] = create_leaf(right)
    else:
        node['right_child'] = best_split(right, loss_func)
        recursive_split(node['right_child'], max_depth, min_records, cur_depth + 1, loss_func)


# Build a decision tree
def build_tree(dataset, max_depth, min_records, loss_func):
	# print("Calling build_tree")
	root_node = best_split(dataset, loss_func)
	recursive_split(root_node, max_depth, min_records, 1, loss_func)
	return root_node


# Predict the output of a data value using the decision tree
def predict_output(node, test_record):
	# print("Calling predict_output")
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


# # print the tree structure
def print_tree(node, depth=0):
	if node['left_child']!=None and node['right_child']!=None:
		print('%s[X%s < %.3f]' % (depth * ' ', (node['column']), node['value']))
		print_tree(node['left_child'], depth + 1)
		print_tree(node['right_child'], depth + 1)
	else:
		print('%s[%s]' % ((depth * ' ', node)))


# Predict the output of a data value using the decision tree
def predict_output(node, test_record):
    # print("Calling predict_output")
    if node['value']==None:
        # # print(node)
        # # print(node['prediction'])
        return node['prediction']
    
    if test_record[node['column']] < node['value']:  # checks if belongs to node's left or right child
        # # print("going to left child")
        return predict_output(node['left_child'], test_record)
        # # print("returning left")
    else:
        # # print("going to right child")
        return predict_output(node['right_child'], test_record)
        # # print("returning right")


# # print the tree structure
#def # print_tree(node, depth=0):
#	if node['left_child']!=None and node['right_child']!=None:
#		# print('%s[X%s < %.3f]' % (depth * ' ', (node['column']), node['value']))
#   else:
#		# print('%s[%s]' % ((depth * ' ', node)))


'''Main Program'''

arguments = sys.argv[1:]
train_dataset=arguments[1]
test_dataset=arguments[3]
min_leaf_size=int(arguments[5])
loss_function=arguments[6][2:]


train_data=pd.read_csv(train_dataset, header=0)
test_data=pd.read_csv(test_dataset, header=0)

print("Starting tree building")
rootnode = build_tree(train_data, 10, min_leaf_size, loss_function)

print("Predicting...")
i=1
output=pd.DataFrame(columns=['Id','output'])
for row in test_data.iterrows():
    output=output.append({'Id':i, 'output':predict_output(rootnode, row[1])}, ignore_index=True)
    i+=1
output['Id']=output['Id'].astype(int)
output.to_csv('output.csv', encoding='utf-8', index=False)

