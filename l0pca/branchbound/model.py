import tensorflow as tf

DTYPE = tf.float32

def gather_by_column(M, row_perms):
    return tf.gather_nd(M, row_perms[:, :, None], batch_dims=1)

class SpcaTraceType(tf.types.experimental.TraceType):

    def __init__(self, spca):
        self.k = spca.k

    def is_subtype_of(self, other):
        return type(other) is SpcaTraceType and self.k == other.k

    def most_specific_common_supertype(self, others):
        return self if all(self == other for other in others) else None

class Spca(object):

    def __init__(self, cov, k):
        self.cov = tf.convert_to_tensor(cov, DTYPE)
        self.cov_perm = tf.argsort(self.cov, direction='DESCENDING', stable=True)
        self.cov_abs = gather_by_column(tf.math.abs(self.cov), self.cov_perm)
        self.cov_2 = gather_by_column(self.cov ** 2, self.cov_perm)
        eigvals, eigvecs = tf.linalg.eigh(self.cov)
        self.eigval = eigvals[-1]
        self.eigvec = eigvecs[:, -1]
        # Frob norm of the entire n-by-n system should upper-bound any bound
        # which we produce. We don't know whether the Gershgorin bound is
        # larger, but then we always take the min of the Gersh and Frob bounds.
        # This is useful when we want a nonnegative, decreasing "priority" from
        # the increasing upper bound.
        self.frobenius_norm = tf.norm(self.cov, 'euclidean')
        self.variance = tf.linalg.diag_part(self.cov)

        self.n = self.cov.shape[0]
        self.k = int(k)

    # def __tf_tracing_type__(self, context):
    #     return SpcaTraceType(self)