import pickle

file_path = "data/filtered_classification_data.pickle"

try:
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    print(f"Type of data: {type(data)}")

    print(f"Keys: {data.keys()}")
    print(f"Reactions: {len(data['reactions'])}")
    print(f"Groups: {len(data['groups'])}")

except Exception as e:
    print(f"Error reading pickle: {e}")
