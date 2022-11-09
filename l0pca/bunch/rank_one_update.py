import tensorflow as tf

def create_update_append_column(
    eig_column_indices, S, eigenvectors, append_column_index, cov, column_norms,
    num_row):
    """Update plan for updating the covariance eigenvalues on insert feature.

    Consider the eigendecomposition as equivalent to the SVD (S^2 and V
    matrices) of the data matrix with a multiplier:
    svd(data / sqrt(num_row - 1))

    This works because:
    cov = (data / sqrt(num_row - 1)).T * (data / sqrt(num_row - 1))

    Then in addition to the eigenvectors (right singular vectors), there are
    also left singular vectors U. Use U.T * v * v.T * U as a similarity
    transformation on the append column v, after this transformation, then the
    symmetric rank-one matrix can be added to the eig column covariance matrix.
    
    v[i] = dot(
        V[:, i],
        cov[append_column_index, eig_column_indices]) / S

    Also, the S^2 matrix (eigenvalues) is the covariance matrix which we get
    if the data columns have the left singular vector U matrix applied to them
    before constructing the Gram matrix (similarity transformation). Therefore,
    we will add v * v.T on top of the diagonal S matrix (this is our symmetric
    rank-one update).

    Finally, to get from n eigenvalues to n+1 eigenvalues, pad the S^2 matrix
    with a zero first row and column (eigh produces eigenvalues in ascending
    order). Augment v concatenated with a new first entry (orthogonal) such that
    the L2 norm of v is equal to column_norm / sqrt(num_row - 1).
    """
    cov_slice = cov[..., append_column_index, :]
    cov_lookup = tf.gather(cov_slice, eig_column_indices)
    update_vec = tf.linalg.matmul(
        batch_matrices.transpose(eigenvectors),
        # Column vector.
        cov_lookup[..., :, None])
    update_vec = update_vec[..., :, 0]
    update_vec /= S
    update_ortho = tf.math.sqrt(
        column_norms[append_column_index] ** 2 / (num_row - 1)
        - tf.linalg.norm(update_vec, axis=-1) ** 2)
    update_vec = tf.concat(
        [
            update_ortho[..., None],
            update_vec,
        ],
        axis=-1,
    )
    return update_vec

def bunch_rational_function_numerator(update_vec):
    return tf.math.square(update_vec)

def bunch_rational_function_denominator(S, column_norm, mu_estimate):
    batch_shape = S.shape[:-1]
    aug_eigenvectors = tf.concat(
        tf.zeros(list(batch_shape) + [1]),
        tf.math.square(S),
        axis=-1,
    )
    return aug_eigenvectors ** 2 / column_norm - mu_estimate[..., None]

def bunch_rational_function_taylor_series(
    update_vec, S, column_norm, mu_estimate, order=0):
    """Taylor series of the function (eig search) w.r.t. mu_estimate."""
    return (
        bunch_rational_function_numerator(update_vec)
        * tf.power(
            bunch_rational_function_denominator(S, column_norm, mu_estimate),
            -(order + 1))
    )