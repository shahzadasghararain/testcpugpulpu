

1. CPU-Focused Task

Problem: Calculating complex mathematical operations, like Fibonacci sequences or prime factorization.
Python
import time

def calculate_fibonacci(n):
    # Standard recursive Fibonacci, CPU-intensive
    if n <= 1:
        return n
    else:
        return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

start_time = time.time()
result = calculate_fibonacci(35) 
end_time = time.time()

print(f"Result: {result}, Time taken: {end_time - start_time} seconds") 


2. GPU-Focused Task

Problem: Training a simple image classification neural network.
Libraries: TensorFlow or PyTorch (these will leverage your available GPU)
Python
import tensorflow as tf 
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Load MNIST dataset (handwritten digits)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Build a simple GPU-accelerated model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax') 
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model (GPU does the heavy lifting)
model.fit(x_train, y_train, epochs=5) 


3. LPU Potential (Conceptual)

Since LPUs are still largely focused on research and niche hardware, there isn't widely available Python support. However, let's imagine a scenario where an LPU is optimized for object detection from low-power camera feeds:
Python
# Simulating interaction with a hypothetical LPU-powered device
import my_lpu_library  

# Assuming continuous feed from camera
while True:
    image_frame = camera.capture_frame()
    objects_detected = my_lpu_library.detect_objects(image_frame) 

    if objects_detected:
        print(f"Objects found: {objects_detected}")
        # Trigger actions based on what's detected
