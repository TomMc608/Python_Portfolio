import os
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Set paths
current_dir = os.path.dirname(os.path.abspath(__file__))
mat_file_path = os.path.join(current_dir, 'mnist-original.mat')
output_dir = os.path.join(current_dir, 'DatasetSplit')

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load the .mat file
if not os.path.exists(mat_file_path):
    raise FileNotFoundError(f"Could not find {mat_file_path}. Please ensure the file is located in {current_dir}.")

mat_data = scipy.io.loadmat(mat_file_path)

# Extract the data and labels
X = mat_data['data'].T  # Transpose to have samples as rows
y = mat_data['label'][0]

# Normalize the dataset
X = X / 255.0

# Split the dataset into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# Shuffle and batch the datasets
batch_size = 32
train_dataset = train_dataset.shuffle(buffer_size=len(X_train)).batch(batch_size)
val_dataset = val_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

# Save the prepared datasets
tf.data.experimental.save(train_dataset, os.path.join(output_dir, 'train_dataset'))
tf.data.experimental.save(val_dataset, os.path.join(output_dir, 'val_dataset'))
tf.data.experimental.save(test_dataset, os.path.join(output_dir, 'test_dataset'))

# Convert TensorFlow datasets to numpy arrays for printing
def dataset_to_numpy_array(dataset):
    return np.concatenate([x.numpy() for x, _ in dataset], axis=0), np.concatenate([y.numpy() for _, y in dataset], axis=0)

X_train_np, y_train_np = dataset_to_numpy_array(train_dataset)
X_val_np, y_val_np = dataset_to_numpy_array(val_dataset)
X_test_np, y_test_np = dataset_to_numpy_array(test_dataset)

# Define verbose variable
verbose = True

# Print the head and end of each dataset
if verbose:
    print("Training Dataset:")
    print("Head:")
    print(X_train_np[:5], y_train_np[:5])
    print("End:")
    print(X_train_np[-5:], y_train_np[-5:])

    print("\nValidation Dataset:")
    print("Head:")
    print(X_val_np[:5], y_val_np[:5])
    print("End:")
    print(X_val_np[-5:], y_val_np[-5:])

    print("\nTest Dataset:")
    print("Head:")
    print(X_test_np[:5], y_test_np[:5])
    print("End:")
    print(X_test_np[-5:], y_test_np[-5:])

    print("Datasets prepared, split, and saved successfully.")


# Delete the .mat file
os.remove(mat_file_path)

print(f"{mat_file_path} has been deleted.")