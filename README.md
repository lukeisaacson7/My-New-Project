# AI Beyond the Robots: The Interactive Neuron Visualizer

Building AI course project — Final Capstone

## Summary

This project proposes an educational web-based simulation tool that visualizes how simple neural network layers process data. It replaces the common "sci-fi robot" media myth with tangible calculations, letting users see exactly how linear combinations and sigmoid functions transform raw data into binary classifications.


## Background

As highlighted in the Building AI conclusion, real-world problems aren't always served in single-serve portions or ready-made templates. My personal motivation for this idea stems from a common issue in tech education:
* **The "Black Box" Problem:** Most beginner AI enthusiasts see neural networks as a black box where data goes in and magic comes out.
* **The Media Misconception:** Public perception of AI is saturated with imagery of metallic human-like androids rather than actual background computer systems and algorithms.
* **The Goal:** To build an interactive gateway into the AI community that bridges the gap between high-level theory and manual mathematical tracking, making the core foundations accessible.


## How is it used?

The solution is an interactive educational dashboard used by students, instructors, or self-learners stepping out of the "safe harbor" of basic courses. 

Users can input custom attributes (such as an animal's height, weight, and length) to see a step-by-step mathematical breakdown of how a hidden layer node evaluates whether the object is a dog or a cat.

For example, if a node receives inputs of `1.3`, `-2.2`, and `9.5` with corresponding weights, the system displays the step-by-step forward propagation:

```python
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def main():
    # 3 Real-world input features
    inputs = [1.3, -2.2, 9.5]
    # Calculated node weights for hidden layer a1
    weights = [-0.76, 0.22, 0.56]
    
    # Step 1: Calculate the linear combination (weighted sum)
    z = sum(i * w for i, w in zip(inputs, weights))
    
    # Step 2: Apply the squashing function for binary probability
    probability = sigmoid(z)
    
    print(f"Linear Combination (Pre-activation): {z:.3f}")
    print(f"Sigmoid Output (Probability Score): {probability:.2f}")

main()
