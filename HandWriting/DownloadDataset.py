import os
import subprocess
import zipfile

# Get the current working directory (where the script is being run)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the path to the kaggle.json file within the current directory
kaggle_json_path = os.path.join(current_dir, 'kaggle.json')

# Ensure the kaggle.json file exists in the current directory
if not os.path.exists(kaggle_json_path):
    raise FileNotFoundError(f"Could not find kaggle.json in {current_dir}. Please place it there.")

# Manually set the environment variables for Kaggle API authentication using the direct path
os.environ['KAGGLE_CONFIG_DIR'] = current_dir

# Define the Kaggle dataset to download
dataset = "avnishnish/mnist-original"

# Define the directory to download the dataset to


# Create the download directory if it doesn't exist
os.makedirs(current_dir, exist_ok=True)

# Download the dataset using the Kaggle API
subprocess.run(["kaggle", "datasets", "download", "-d", dataset, "-p", current_dir])

# Define the name of the downloaded zip file (typically the dataset name with .zip)
zip_file = os.path.join(current_dir, dataset.split("/")[-1] + ".zip")

# Unzip the file if it exists
if os.path.exists(zip_file):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(current_dir)
    os.remove(zip_file)  # Delete the zip file

print("Dataset downloaded and extracted successfully.")