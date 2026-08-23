import numpy as np


def destination_score(distances_m, attractiveness, beta_distance, gamma_attractiveness):
    """
    Destination ranking shared by Continuous and H3 preprocessing.
    """
    return (
        -beta_distance * distances_m
        + gamma_attractiveness * attractiveness
    )
