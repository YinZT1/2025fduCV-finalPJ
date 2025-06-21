import json
import os
import argparse

def split_transforms(input_path):
    """
    Reads a single transforms.json file and splits it into
    transforms_train.json and transforms_test.json, holding out
    every 8th frame for the test set.
    """
    print(f"Loading transforms from {input_path}")
    with open(input_path, 'r') as f:
        data = json.load(f)

    # The directory where the input file is located
    output_dir = os.path.dirname(input_path)

    # Copy metadata to both train and test splits
    train_data = data.copy()
    test_data = data.copy()

    train_data['frames'] = []
    test_data['frames'] = []
    
    # Define how to split the data
    hold_every = 8

    print(f"Splitting frames... Every {hold_every}-th frame will be in the test set.")
    
    for i, frame in enumerate(data['frames']):
        if i % hold_every == 0:
            test_data['frames'].append(frame)
        else:
            train_data['frames'].append(frame)
            
    # Define output file paths
    train_output_path = os.path.join(output_dir, 'transforms_train.json')
    test_output_path = os.path.join(output_dir, 'transforms_test.json')

    print(f"Writing {len(train_data['frames'])} frames to {train_output_path}")
    with open(train_output_path, 'w') as f:
        json.dump(train_data, f, indent=2)

    print(f"Writing {len(test_data['frames'])} frames to {test_output_path}")
    with open(test_output_path, 'w') as f:
        json.dump(test_data, f, indent=2)
        
    print("\nSplit complete!")


if __name__ == "__main__":    
    split_transforms(r'/remote-home/yinzhitao/TensoRF/f_data/transforms.json')