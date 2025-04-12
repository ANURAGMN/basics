import numpy as np

x=np.array([
    [0,0],
    [0,1],
    [1,1],
    [1,0],

])
y=np.array([0,0,0,1])

weights=np.zeros(2)
bias=0

lr=0.1
def step_function(x):
    return 1 if x> 0 else 0
epochs = 10
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}")
    for i in range(len(x)):
        z = np.dot(x[i], weights) + bias
        prediction = step_function(z)
        error=y[i]-prediction
        weights += lr * error * x[i]
        bias += lr * error

        print(f"  Input: {x[i]}, Predicted: {prediction}, Actual: {y[i]}")
        print(f"  Updated Weights: {weights}, Bias: {bias}")
    print("-" * 50)
print("Testing the trained perceptron:")
for i in range(len(x)):
    z = np.dot(x[i], weights) + bias
    prediction = step_function(z)
    print(f"Input: {x[i]} => Prediction: {prediction}")
