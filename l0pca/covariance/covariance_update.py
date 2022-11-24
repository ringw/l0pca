from l0pca.util import batch_matrices
import tensorflow as tf

def update_vector(
    covariance,
    current_column_index,
    current_eigenvalues,
    current_eigenvectors,
    update_column_index):
    batch_dims = len(covariance.shape) - 2
    cov_slice = covariance[..., update_column_index, :]
    append_cov = tf.gather(cov_slice, current_column_index, batch_dims=batch_dims)
    update_vec = tf.linalg.matmul(
        current_eigenvectors,
        append_cov[..., :, None],
        adjoint_a=True,
    )
    update_vec = update_vec[..., :, 0]
    update_vec *= tf.where(
        current_eigenvalues > 0,
        1. / tf.sqrt(current_eigenvalues),
        0)
    update_ortho_squared = (
        covariance[update_column_index, update_column_index]
            - tf.linalg.norm(update_vec, axis=-1) ** 2
    )
    assert_ortho = tf.debugging.assert_non_negative(update_ortho_squared)
    with tf.control_dependencies([assert_ortho]):
        update_vec = tf.concat(
            [
                update_vec,
                # update_vec is a flat vector, so we only insert one dim for this
                # scalar value.
                tf.math.sqrt(update_ortho_squared)[..., None],
            ],
            axis=-1,
        )
        return update_vec