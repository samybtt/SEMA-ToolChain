import subprocess

methods = ["WSELECT", "WSELECT2", "WSELECTSET2", "CDFS", "CBFS", "CSTOCH", "CSTOCH2", "CSTOCHSET2", "STOCH"]
classifiers = ["wl", "gin", "ginjk"]
num_layers = [2, 3, 4, 5, 6]
splits = ["split_0", "split_1", "split_2", "split_3", "split_4"]

# with open(f"output/gnn_eval/ml_eval_stats.csv", "w") as f:
#     f.write("iter,method,classifier,num_layers,split,accuracy,balanced_accuracy,precision,recall,f1_score\n")

for method in methods:
    classifier = "wl"
    for split in splits:
        print(f"############# Training {method} with {classifier} on {split} #############")
        # Construct the train_command
        train_command = f"python ToolChainClassifier/ToolChainClassifier.py --train --classifier={classifier} ./databases/examples_samy/big_dataset/merged/traindatabal/{method}/{split}"
        # Execute the train_command
        subprocess.run(train_command, shell=True)
        num_layer = 0
        itr = 0
        with open(f"output/gnn_eval/ml_eval_stats.csv", "a") as f:
            f.write(f"{itr},{method},{classifier},{num_layer},{split},")
        print(f"############# Testing {method} with {classifier} on {split} #############")
        # Construct the test_command
        test_command = f"python ToolChainClassifier/ToolChainClassifier.py --classifier={classifier} ./databases/examples_samy/big_dataset/merged/testdatabal/{method}/{split}"
        # Execute the test_command
        subprocess.run(test_command, shell=True)

# Iterate over different parameters
for num_layer in num_layers:
    for method in methods:
        for classifier in ["gin", "ginjk"]:
            for split in splits:
                for itr in range(5):
                    print(f"############# Training {method} with {classifier} and {num_layer} layers on {split} #############")
                    # Construct the train_command
                    train_command = f"python ToolChainClassifier/ToolChainClassifier.py --train --classifier={classifier} --num_layers={num_layer} ./databases/examples_samy/big_dataset/merged/traindatabal/{method}/{split}"
                    # Execute the train_command
                    subprocess.run(train_command, shell=True)
                    with open(f"output/gnn_eval/ml_eval_stats.csv", "a") as f:
                        f.write(f"{itr},{method},{classifier},{num_layer},{split},")
                    print(f"############# Testing {method} with {classifier} and {num_layer} layers on {split} #############")
                    # Construct the test_command
                    test_command = f"python ToolChainClassifier/ToolChainClassifier.py --classifier={classifier} ./databases/examples_samy/big_dataset/merged/testdatabal/{method}/{split}"
                    # Execute the test_command
                    subprocess.run(test_command, shell=True)
                


