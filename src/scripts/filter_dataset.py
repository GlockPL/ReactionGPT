import pickle
from pathlib import Path

import numpy as np

from src.utils import get_project_root


def filter_dataset(min_items=30):
    with Path(get_project_root() / "data" / "bertaction_corrected_catals.pickle").open(
        "rb"
    ) as handle:
        data = pickle.load(handle)

    print(f"Original keys: {data.keys()}")
    react = data["reactions"]
    group = data["groups"]
    y_raw = np.array(data["y"])

    # Handle one-hot encoding if present
    if y_raw.ndim > 1:
        y = np.argmax(y_raw, axis=1)
    else:
        y = y_raw

    # Count occurrences of each class
    uq, cnt = np.unique(y, return_counts=True)

    # Identify classes that meet the threshold
    valid_classes = uq[cnt >= min_items]
    print(f"Found {len(valid_classes)} classes with >= {min_items} samples.")

    # Create mapping from old class index to new contiguous index (0 to N-1)
    old_to_new_map = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_classes)}
    valid_classes_set = set(valid_classes)

    filtered_reactions = []
    filtered_groups = []
    filtered_y_indices = []

    # Filter data
    print("Filtering data...")
    for i in range(len(y)):
        if y[i] in valid_classes_set:
            filtered_reactions.append(react[i])
            filtered_groups.append(group[i])
            filtered_y_indices.append(old_to_new_map[y[i]])

    print(f"Kept {len(filtered_reactions)} samples out of {len(y)}.")

    # Reconstruct one-hot encoded y for the filtered data
    num_classes = len(valid_classes)
    filtered_y = np.zeros((len(filtered_y_indices), num_classes))
    for i, class_idx in enumerate(filtered_y_indices):
        filtered_y[i, class_idx] = 1

    # Update metadata maps with new indices
    new_catal_num_to_smiles_map = {}
    if "catal_num_to_smiles_map" in data:
        for old_idx, new_idx in old_to_new_map.items():
            if old_idx in data["catal_num_to_smiles_map"]:
                new_catal_num_to_smiles_map[new_idx] = data["catal_num_to_smiles_map"][
                    old_idx
                ]

    new_catal_smiles_to_num_map = {}
    if "catal_smiles_to_num_map" in data:
        for smiles, old_idx in data["catal_smiles_to_num_map"].items():
            if old_idx in old_to_new_map:
                new_catal_smiles_to_num_map[smiles] = old_to_new_map[old_idx]

    filtered_data = {
        "reactions": filtered_reactions,
        "groups": filtered_groups,
        "y": filtered_y.tolist() if isinstance(data["y"], list) else filtered_y,
        "catal_num_to_smiles_map": new_catal_num_to_smiles_map,
        "catal_smiles_to_num_map": new_catal_smiles_to_num_map,
        "num_classes": num_classes,
    }

    output_path = get_project_root() / "data" / "filtered_classification_data.pickle"
    with output_path.open("wb") as handle:
        pickle.dump(filtered_data, handle)

    print(f"Filtered dataset saved to {output_path}")


if __name__ == "__main__":
    filter_dataset()
