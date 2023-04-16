"""
Given a pre-generated dataset, we can take a subset and save that as a smaller pre-generated dataset
"""
from reusable.util import load_args, save_datasets, load_datasets, gen_file_name


code = 19
count = 1
args_file_name = "1"


args = load_args(code, count, args_file_name)

train, test = load_datasets(code, gen_file_name(code, args, "raw_gp", True), on_arc=True)

old_test = args["batch_size"] * args["test_num_batches"]
old_train = args["batch_size"] * args["train_num_batches"]

print(train.shape, old_train)

test = test.reshape((old_test, -1))
train = train.reshape((old_train, -1))

# change params

args["test_num_batches"] = 2
args["train_num_batches"] = 200
args["batch_size"] = 400


new_test = args["batch_size"] * args["test_num_batches"]
new_train = args["batch_size"] * args["train_num_batches"]

assert new_test <= old_test and new_train <= old_train

test = test[:new_test, :]
train = train[:new_train, :]

test = test.reshape((1, args["test_num_batches"] * args["batch_size"], -1))
train = train.reshape((args["train_num_batches"], args["batch_size"], -1))

save_datasets(code, gen_file_name(code, args, "raw_gp", True), train, test)
