import os
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Set paths
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(current_dir, 'DatasetSplit')
model_dir = os.path.join(current_dir, 'saved_model')

# Ensure directories exist
os.makedirs(model_dir, exist_ok=True)

# Load datasets
print(f"Loading training dataset from: {os.path.join(dataset_dir, 'train_dataset')}")
print(f"Loading validation dataset from: {os.path.join(dataset_dir, 'val_dataset')}")
print(f"Loading test dataset from: {os.path.join(dataset_dir, 'test_dataset')}")

train_dataset = tf.data.Dataset.load(os.path.join(dataset_dir, 'train_dataset'))
val_dataset = tf.data.Dataset.load(os.path.join(dataset_dir, 'val_dataset'))
test_dataset = tf.data.Dataset.load(os.path.join(dataset_dir, 'test_dataset'))

# For debugging, use the dataset directly without augmentation
augmented_train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Define a simple feedforward neural network model
def create_simple_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# Create model instance
model = create_simple_model(input_shape=(784,), num_classes=10)

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Inspect dataset shapes
for x_batch, y_batch in augmented_train_dataset.take(1):
    print(f"Batch X shape: {x_batch.shape}")
    print(f"Batch Y shape: {y_batch.shape}")

# Custom training loop to print epoch number and ETA on the same line
epochs = 10
start_time = time.time()

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}", end="")
    model.fit(
        augmented_train_dataset,
        validation_data=val_dataset,
        epochs=1,
        verbose=0  # Set to 0 to suppress the default progress bar
    )
    elapsed_time = time.time() - start_time
    eta = elapsed_time / (epoch + 1) * (epochs - (epoch + 1))
    eta_str = time.strftime('%H:%M:%S', time.gmtime(eta))
    print(f"\rEpoch {epoch + 1}/{epochs} - ETA: {eta_str}", end="")
    if epoch < epochs - 1:
        print(end="\r")
    else:
        print()  # Ensure we move to the next line after the final epoch

# Evaluate the model on test dataset
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Save the trained model in Keras native format
model.save(os.path.join(model_dir, 'mnist_simple_model.keras'))
print("Model saved successfully.")
