"""The losses, the brain of the model"""
import math

def loss_few_neighbours(x: float, freq: float, approx_kliek_getal: float) -> float:
    """
    Favor nodes that have a lot of edges

    keyword arguments:
    x -- probability of node between 0 and 1
    freq -- amount of edges with node
    approx_kliek_getal -- approx of the kliek getal
    """
    if freq < approx_kliek_getal:
        return 1
    else:
        return 1 - x

def loss_neighbors_close(x: float, y: float) -> float:
    """
    Takes as input the probability of 2 neighbours,
    bigger loss if any of both is small, small output if both high

    keyword arguments:
    x -- neighbor 1 probability between 0 and 1
    y -- neighbor 2
    """
    return (1 - x) * (1 - y) + abs(x - y)

def loss_non_neighbors_far(x: float, y: float) -> float:
    """
    Takes as input the probability of 2 neighbours,
    bigger loss if both are high, small output in all other cases

    keyword arguments:
    x -- neighbor 1 probability between 0 and 1
    y -- neighbor 2
    p -- loss param for how heavy we make the punish
    """

    return math.exp((x + y - 2))

def loss_grade(x: float, grade: int, n: int) -> float:
    """
    Promote nodes with high grade if their probability is high?

    keyword arguments:
    x -- node probability between 0 and 1
    grade -- amount of neighbors of node x
    n -- total amount of nodes
    """

    return ((n - grade)/n)*x
