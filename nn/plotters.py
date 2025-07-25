"""Plotter collection"""
import numpy as np
import matplotlib.pyplot as plt
from nn.losses import loss_few_neighbours, loss_neighbors_close, loss_non_neighbors_far

def plot_loss():
    """Plots the loss functions"""
    # generate x y mesh
    x_values = np.arange(0, 1.01, 0.01)
    y_values = np.arange(0, 1.01, 0.01)
    X, Y = np.meshgrid(x_values, y_values)

    # NEIGHBORS
    Z = np.vectorize(loss_neighbors_close)(X, Y)
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X, Y, Z, cmap='viridis')
    plt.colorbar(cp)
    plt.title('Neighbors keep neighbors high')
    plt.xlabel('Probability of Neighbor 1')
    plt.ylabel('Probability of Neighbor 2')
    plt.show()

    # NON NEIGHBORS
    Z = np.vectorize(loss_non_neighbors_far)(X, Y)
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X, Y, Z, cmap='viridis')
    plt.colorbar(cp)
    plt.title('Non-Neighbors should keep eachother low')
    plt.xlabel('Probability of Neighbor 1')
    plt.ylabel('Probability of Neighbor 2')
    plt.show()

    # PUNISHING DEGREE LOWER THAN KLIEKHEURISTIC
    kliek_value = 2
    x_values = np.arange(0, 1.01, 0.01)
    y_values = np.arange(0, 5.01, 0.01)
    x_mesh, y_mesh = np.meshgrid(x_values, y_values)
    z_values = np.vectorize(loss_few_neighbours)(x_mesh, y_mesh, kliek_value)
    print(z_values)
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(x_mesh, y_mesh, z_values, cmap='viridis')
    plt.colorbar(cp)
    plt.title('Fewer neighbors than kliek heuristic needs to be punished')
    plt.xlabel('Probability of node')
    plt.axhline(y=kliek_value, color='r', linestyle='--', label=f'Kliek getal = {kliek_value}')
    plt.ylabel('Amount of neighbors')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_loss()
    x_values = np.arange(0, 1.01, 0.01)
    y_value = 1
    kliek = 1
    z_values = [loss_few_neighbours(x, y_value, kliek) for x in x_values]
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, z_values, label='loss_few_neighbours(x, 1, 1)')
    plt.title('Loss for few neighbors with fixed y=1 and kliek=1')
    plt.xlabel('Probability of node')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
