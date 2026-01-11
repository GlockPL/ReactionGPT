import pickle
import random
from pathlib import Path

import numpy as np
from rdkit import Chem

from src.utils import get_project_root


def enumerate_reaction_smiles(reaction_smiles, n_variants=5):
    """
    Enumerates a reaction SMILES string by randomizing each component.
    Format: reactant.reactant>>product.product
    """
    # Split the reaction into its main parts (Reactants>>Products)
    parts = reaction_smiles.split(">>")
    if len(parts) < 2:
        # Fallback or invalid format
        return [reaction_smiles]

    # Function to enumerate a component that might have multiple molecules (separated by '.')
    def enumerate_component(component_smiles):
        if not component_smiles:
            return [""]

        # Split into individual molecules
        mols = component_smiles.split(".")
        enumerated_mols_lists = []

        for m in mols:
            mol_obj = Chem.MolFromSmiles(m)
            if not mol_obj:
                enumerated_mols_lists.append([m])
                continue

            # Generate variants for this specific molecule
            variants = {Chem.MolToSmiles(mol_obj, canonical=True)}
            attempts = 0
            # Try to generate n_variants distinct SMILES
            while len(variants) < n_variants and attempts < n_variants * 5:
                variants.add(Chem.MolToSmiles(mol_obj, doRandom=True, canonical=False))
                attempts += 1
            enumerated_mols_lists.append(list(variants))

        # Create full component strings by picking one variant of each molecule
        # We shuffle the order of molecules in the '.' string too for extra variety
        combined_variants = []
        for _ in range(n_variants):
            current_combination = [
                random.choice(v_list) for v_list in enumerated_mols_lists
            ]
            random.shuffle(current_combination)  # A.B is the same as B.A
            combined_variants.append(".".join(current_combination))

        return combined_variants

    # Enumerate each part
    enum_reactants = enumerate_component(parts[0])
    enum_products = enumerate_component(parts[1])

    # Combine parts back into reaction strings
    reaction_variants = set()
    for i in range(n_variants):
        r = enum_reactants[i % len(enum_reactants)]
        p = enum_products[i % len(enum_products)]
        reaction_variants.add(f"{r}>>{p}")

    return list(reaction_variants)


def balance_dataset():
    input_path = get_project_root() / "data" / "filtered_classification_data.pickle"
    output_path = get_project_root() / "data" / "balanced_classification_data.pickle"

    print(f"Loading data from {input_path}...")
    with input_path.open("rb") as handle:
        data = pickle.load(handle)

    reactions = data["reactions"]
    groups = data["groups"]
    y_raw = np.array(data["y"])

    # Handle one-hot encoding if present
    if y_raw.ndim > 1:
        y_indices = np.argmax(y_raw, axis=1)
    else:
        y_indices = y_raw

    # Determine class counts
    unique_classes, counts = np.unique(y_indices, return_counts=True)
    target_count = np.max(counts)
    print(f"Max class count is {target_count}. Balancing all classes to this count.")

    # Organize data by class
    # Storing tuple: (reaction_smiles, group_id, original_index)
    class_data = {c: [] for c in unique_classes}
    for idx, (r, g, label) in enumerate(zip(reactions, groups, y_indices)):
        class_data[label].append((r, g, idx))

    balanced_reactions = []
    balanced_groups = []
    balanced_original_indices = []
    balanced_y_indices = []

    for label in unique_classes:
        samples = class_data[label]
        n_current = len(samples)
        n_needed = target_count - n_current

        # 1. Add all original samples
        for r, g, orig_idx in samples:
            balanced_reactions.append(r)
            balanced_groups.append(g)
            balanced_original_indices.append(orig_idx)
            balanced_y_indices.append(label)

        # 2. Augment if needed
        if n_needed > 0:
            print(
                f"Class {label}: Augmenting {n_needed} samples (original: {n_current})"
            )
            generated_count = 0
            
            # Shuffle source pool to pick random samples to augment
            source_pool = list(samples)
            random.shuffle(source_pool)
            
            pool_idx = 0
            while generated_count < n_needed:
                # Cycle through the source pool
                if pool_idx >= len(source_pool):
                    random.shuffle(source_pool)
                    pool_idx = 0
                
                r_src, g_src, orig_idx_src = source_pool[pool_idx]
                pool_idx += 1

                # Generate variants
                # Requesting a small batch of variants
                variants = enumerate_reaction_smiles(r_src, n_variants=5)
                
                for v in variants:
                    if generated_count >= n_needed:
                        break
                    
                    # We accept the variant even if it's identical to source to satisfy the count,
                    # but typically enumerate_reaction_smiles produces randomized SMILES.
                    balanced_reactions.append(v)
                    balanced_groups.append(g_src)
                    # CRITICAL: Use the SAME original index to prevent leakage
                    balanced_original_indices.append(orig_idx_src)
                    balanced_y_indices.append(label)
                    
                    generated_count += 1
        else:
            print(f"Class {label}: Already at max count {n_current}")

    # Shuffle the final balanced dataset
    print("Shuffling balanced dataset...")
    combined = list(
        zip(
            balanced_reactions,
            balanced_groups,
            balanced_original_indices,
            balanced_y_indices,
        )
    )
    random.shuffle(combined)
    (
        balanced_reactions,
        balanced_groups,
        balanced_original_indices,
        balanced_y_indices,
    ) = zip(*combined)

    # Convert y back to one-hot format
    num_classes = data["num_classes"]
    balanced_y = np.zeros((len(balanced_y_indices), num_classes))
    for i, label in enumerate(balanced_y_indices):
        balanced_y[i, int(label)] = 1

    # Save the result
    balanced_data = {
        "reactions": list(balanced_reactions),
        "groups": list(balanced_groups),
        "y": balanced_y.tolist(),
        "original_indices": list(balanced_original_indices),  # New field for leakage prevention
        "catal_num_to_smiles_map": data.get("catal_num_to_smiles_map", {}),
        "catal_smiles_to_num_map": data.get("catal_smiles_to_num_map", {}),
        "num_classes": num_classes,
    }

    with output_path.open("wb") as handle:
        pickle.dump(balanced_data, handle)

    print(
        f"Saved balanced dataset to {output_path}. Total samples: {len(balanced_reactions)}"
    )


if __name__ == "__main__":
    balance_dataset()
