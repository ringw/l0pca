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
        self.frobenius_norm = tf.norm(self.cov, 'euclidean')

        self.n = self.cov.shape[0]
        self.k = int(k)

    # def __tf_tracing_type__(self, context):
    #     return SpcaTraceType(self)