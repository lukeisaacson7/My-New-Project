# Demystifying the Neuron: Beyond the AI Media Myth

Final project for the Building AI course

## Summary

This project explores the mathematical reality of neural networks, shifting focus from sci-fi media tropes to actual algorithmic mechanics. By breaking down and manually calculating forward propagation (weights, linear combinations, and sigmoid activations), it demonstrates how simple functions drive complex classifications.


## Background

When people search for "AI," online results are flooded with images of glowing human-like robots or metallic brains. This creates a massive misconception about what artificial intelligence actually is. 
* **The Problem:** The public view of AI is disconnected from the reality of data science, leading to either inflated fear or unrealistic expectations.
* **Motivation:** My goal is to ground this topic by demonstrating that AI isn't magic or a sentient robot—it's foundational math, background computer systems, and data processing.
* **Importance:** Understanding the building blocks (like basic regressions and activation functions) is essential for anyone wanting to responsibly build, critique, or use modern technology.


## How is it used?

This project serves as an educational framework and a proof-of-concept for manual neural network tracking. It is designed for students, educators, or tech enthusiasts who want to look inside the "black box" of a hidden layer node.

For example, when a network needs to classify data—such as predicting whether an animal is a dog or a cat based on height, weight, and length—this model shows exactly how raw features are transformed into actual probabilities.

Here is a look at the simple 3-layer architecture being modeled:
![Neural Network Diagram](https://images.unsplash.com/photo-1527474305487-b87b222841cc?q=80&w=600&auto=format&fit=crop)

### Example: Manual Forward Propagation Challenge
Below is the core Python logic used to replicate the manual calculation I performed on a hidden layer node ($a_1$) processing three distinct inputs:

```python
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def main():
    # 3 Input nodes
    inputs = [1.3, -2.2, 9.5]
    # Weights connecting to hidden node a1
    weights = [-0.76, 0.22, 0.56]
    
    # 1. Calculate the linear combination (weighted sum)
    linear_combination = sum(i * w for i, w in zip(inputs, weights))
    
    # 2. Pass through the activation function
    identity_output = linear_combination
    probability_output = sigmoid(linear_combination)
    
    print(f"Linear Combination (Pre-activation): {linear_combination:.3f}")
    print(f"Output with Identity Function: {identity_output:.3f}")
    print(f"Output with Sigmoid Function (Probability): {probability_output:.2f}")

if __name__ == "__main__":
    main()
