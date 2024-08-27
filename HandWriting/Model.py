import os
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers.schedules import ExponentialDecay

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

# Data augmentation pipeline
data_augmentation = keras.Sequential([
    layers.Reshape((28, 28, 1)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomFlip(mode='horizontal'),
    layers.Reshape((784,))
])

# Apply data augmentation to the training dataset
augmented_train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x), y))
augmented_train_dataset = augmented_train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Define a deeper and more complex model with CNNs and additional layers
def create_deep_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    x = layers.Reshape((28, 28, 1))(inputs)
    
    # Convolutional Layers
    x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, kernel_size=(3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)

    # Flatten and Dense Layers
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# List of optimizers to try
optimizers = {
    "AdamW": keras.optimizers.AdamW(learning_rate=ExponentialDecay(0.001, decay_steps=100000, decay_rate=0.96)),
    "Nadam": keras.optimizers.Nadam(learning_rate=ExponentialDecay(0.001, decay_steps=100000, decay_rate=0.96)),
    "RMSprop": keras.optimizers.RMSprop(learning_rate=ExponentialDecay(0.001, decay_steps=100000, decay_rate=0.96))
}

best_accuracy = 0
best_optimizer = None

for opt_name, optimizer in optimizers.items():
    print(f"\nTesting optimizer: {opt_name}")
    
    # Create a new model instance for each optimizer
    model = create_deep_model(input_shape=(784,), num_classes=10)
    
    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    # Train the model
    epochs = 20
    start_time = time.time()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}", end="")
        model.fit(
            augmented_train_dataset,
            validation_data=val_dataset,
            epochs=1,
            verbose=0
        )
        elapsed_time = time.time() - start_time
        eta = elapsed_time / (epoch + 1) * (epochs - (epoch + 1))
        eta_str = time.strftime('%H:%M:%S', time.gmtime(eta))
        print(f"\rEpoch {epoch + 1}/{epochs} - ETA: {eta_str}", end="")
        if epoch < epochs - 1:
            print(end="\r")
        else:
            print()
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_optimizer = opt_name
    
    # Save the model for this optimizer
    model.save(os.path.join(model_dir, f'mnist_deep_model_{opt_name}.keras'))

print(f"\nBest optimizer: {best_optimizer} with accuracy: {best_accuracy:.4f}")

# Finalize by saving the best model
print("Training completed and models saved.")
