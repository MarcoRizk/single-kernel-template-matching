import os

import yaml


def save_dimensions_to_yaml(h, w, d, file_path="kernelconfig.yaml"):
    # Create a dictionary with dimensions
    data = {
        "height": h,
        "width": w,
        "depth": d
    }

    # Write the dictionary to a YAML file
    with open(file_path, "w") as file:
        yaml.dump(data, file)
    print(f"Dimensions saved to {file_path}")


def load_dimensions_from_yaml(file_path="dimensions.yaml"):
    # Read the dictionary from the YAML file
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)

    # Extract dimensions from the loaded data
    h = data.get("height")
    w = data.get("width")
    d = data.get("depth")

    print(f"Dimensions loaded from {file_path}")
    return h, w, d


def save_image_with_suffix(img, original_path, suffix="-kernel-matched"):
    # Split the file path into directory, base name, and extension
    directory, filename = os.path.split(original_path)
    basename, ext = os.path.splitext(filename)

    # Create a new filename with the suffix
    new_filename = f"{basename}{suffix}{ext}"
    new_file_path = os.path.join(directory, new_filename)

    # Save the image to the new file path
    img.save(new_file_path)
    print(f"Image saved as {new_file_path}")
