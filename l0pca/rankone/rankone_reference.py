"""Naive rank-one update.

This applies the symmetric update to the matrix, and uses linalg libraries
naively. Therefore, this is a base library to benchmark against rank-one
improvements.
"""
import tensorflow as tf

def rankone_update_eigenvalues(eig, v):
    diag_matrix = tf.linalg.diag(eig)
    matrix = diag_matrix + tf.linalg.matmul(
        v[..., :, None], v[..., None, :])
    return tf.linalg.eigh(matrix)[0]