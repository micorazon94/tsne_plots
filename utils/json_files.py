import json
import os

def get_class_labels(class_indices_list, json_path='data/cat_to_name.json'):
    """
    Given a list of class indices, return the corresponding class names.

    Parameters:
    - class_indices: list of int, the class indices for which to find labels.
    - json_path: str, path to the JSON file containing the class-to-name mapping.

    Returns:
    - class_labels: list of str, the class names corresponding to the input indices.
    """

    # Check if the path is relative; if so, convert to an absolute path
    if not os.path.isabs(json_path):
        json_path = os.path.abspath(json_path)
        print(f"Absolute JSON path: {json_path}")

    # Load the class-to-name mapping from the JSON file
    with open(json_path, 'r') as f:
        class_to_name = json.load(f)

    # Extract class names for the given indices
    class_labels_list = [class_to_name[str(index)] for index in class_indices_list if str(index) in class_to_name]

    return class_labels_list


if __name__ == "__main__":
    # Example usage

    class_indices_list = [1, 88]
    class_labels = get_class_labels(class_indices_list)
    print(class_labels)  # Output: ['Class Name for 1', 'Class Name for 88']