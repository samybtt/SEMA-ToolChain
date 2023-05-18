import os
import random
import shutil

# Set the directory paths
data_dir = "databases/examples_samy/big_dataset/merged/alldata"
train_dir = "databases/examples_samy/big_dataset/merged/traindatabal"
test_dir = "databases/examples_samy/big_dataset/merged/testdatabal"
possible_families = ["bancteian", "ircbot", "sillyp2p", "sytro", "simbot", "RemcosRAT", "delf", "nitol", "gandcrab", "wabot"]
# possible_families = ["bancteian", "ircbot", "sillyp2p", "sytro", "simbot", "FeakerStealer", "sfone", "lamer", "RedLineStealer", "RemcosRAT", "delf", "nitol", "gandcrab", "wabot"]


# Set the train/test split ratio (e.g., 80/20 split)
split_ratio = 0.7


for method in os.listdir(data_dir):
    method_dir = os.path.join(data_dir, method)
    # Loop over each family directory in the data directory
    train_method_dir = os.path.join(train_dir, method)
    test_method_dir = os.path.join(test_dir, method)
    os.makedirs(train_method_dir, exist_ok=True)
    os.makedirs(test_method_dir, exist_ok=True)
    for family_dir in os.listdir(method_dir):
        if family_dir not in possible_families:
            continue
        # Create the corresponding directories in the train/test directory
        train_family_dir = os.path.join(train_method_dir, family_dir)
        test_family_dir = os.path.join(test_method_dir, family_dir)
        os.makedirs(train_family_dir, exist_ok=True)
        os.makedirs(test_family_dir, exist_ok=True)

        # Get the list of all GS files in the family directory
        gs_files = os.listdir(os.path.join(method_dir, family_dir))
        
        # Shuffle the list of GS files randomly
        random.shuffle(gs_files)

        # # Split the list of GS files into train and test sets
        # split_idx = int(len(gs_files) * split_ratio)
        # split_idx = int(36 * split_ratio)
        # train_files = gs_files[:split_idx]
        # test_files = gs_files[split_idx:]

        # Split the list of GS files into train and test sets
        # split_idx = int(len(gs_files) * split_ratio)
        train_files = gs_files[:26]
        test_files = gs_files[26:36]

        # Move the train files to the train directory and the test files to the test directory
        for train_file in train_files:
            src = os.path.join(method_dir, family_dir, train_file)
            dst = os.path.join(train_family_dir, train_file)
            shutil.copy(src, dst)

        for test_file in test_files:
            src = os.path.join(method_dir, family_dir, test_file)
            dst = os.path.join(test_family_dir, test_file)
            shutil.copy(src, dst)
