import h5py
import numpy as np
import tensorflow as tf

import model
import node
import partial_solution
import search
import top_k

f = h5py.File("../Optimal-SPCA/data/communities.jld")
cov = np.asarray(f["normCommunities"])
n = cov.shape[0]
k = 5

spca = model.Spca(cov, k)

y = np.full(n, -1)
y[[31]] = 1
y = tf.constant(y)

# print(partial_solution.solve_pseudovec(spca, y))
# print(top_k.top_k(spca.cov_abs, spca.cov_perm, k, y))
#print(bounds.bound_spca(spca, y))
# print(node.process_node(spca, node.build_root(spca)))

q = search.new_search_queue(spca)
best_obj = tf.constant(0., model.DTYPE)
best_y = tf.zeros(spca.n, model.DTYPE)
for i in range(2000):
    q, best_obj, best_y = search.step(spca, q, best_obj, best_y)
print(q.dequeue())
print(q.size())
print(best_obj)
