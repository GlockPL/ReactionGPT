import pickle
from pathlib import Path

import numpy as np

from src.utils import get_project_root


def balance_dataset(min_items=200):
    with Path(get_project_root() / "data" / "bertaction_corrected_catals.pickle").open(
        "rb"
    ) as handle:
        data = pickle.load(handle)

    print(data.keys())
    react = data["reactions"]
    group = data["groups"]
    y = np.array(data["y"])
    y = np.argmax(y, axis=1)

    max_class = np.max(y)

    class_data_dict = {i: [] for i in range(max_class + 1)}
    for i in range(len(y)):
        class_data_dict[y[i]].append((react[i], group[i]))

    uq, cnt = np.unique(y, return_counts=True)
    bigger_than_min_items = uq[cnt > min_items]
    dataset_size = np.min(cnt[bigger_than_min_items])

    print(
        f"Sampling {dataset_size} items from each of the {len(bigger_than_min_items)} classes."
    )

    # Map old class indices to new contiguous indices (0 to N-1)
    old_to_new_map = {
        old_idx: new_idx for new_idx, old_idx in enumerate(bigger_than_min_items)
    }

    balanced_reactions = []
    balanced_groups = []
    balanced_y_indices = []

    for class_idx in bigger_than_min_items:
        samples = class_data_dict[class_idx]
        # Randomly sample dataset_size items from the current class
        sampled_indices = np.random.choice(len(samples), dataset_size, replace=False)
        for idx in sampled_indices:
            r, g = samples[idx]
            balanced_reactions.append(r)
            balanced_groups.append(g)
            # Use the new reindexed class index
            balanced_y_indices.append(old_to_new_map[class_idx])

    # Reconstruct one-hot encoded y using the new number of classes
    num_classes = len(bigger_than_min_items)
    balanced_y = np.zeros((len(balanced_y_indices), num_classes))
    for i, class_idx in enumerate(balanced_y_indices):
        balanced_y[i, class_idx] = 1

    # Shuffle the balanced dataset
    indices = np.arange(len(balanced_reactions))
    np.random.shuffle(indices)

    balanced_reactions = [balanced_reactions[i] for i in indices]
    balanced_groups = [balanced_groups[i] for i in indices]
    balanced_y = balanced_y[indices]

    # Reindex class maps (names)
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

    balanced_data = {
        "reactions": balanced_reactions,
        "groups": balanced_groups,
        "y": balanced_y.tolist() if isinstance(data["y"], list) else balanced_y,
        "catal_num_to_smiles_map": new_catal_num_to_smiles_map,
        "catal_smiles_to_num_map": new_catal_smiles_to_num_map,
        "num_classes": num_classes,
    }

    output_path = get_project_root() / "data" / "balanced_classification_data.pickle"
    with output_path.open("wb") as handle:
        pickle.dump(balanced_data, handle)

    print(f"Balanced dataset saved to {output_path}")
    print(f"Total samples: {len(balanced_reactions)}")


if __name__ == "__main__":
    balance_dataset()
