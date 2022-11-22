# Evaluating Sparse PCA With ℓ<sub>0</sub> Constraint

Sparse PCA with ℓ<sub>0</sub> constraint is a combinatorial optimization problem, projecting the data matrix onto a subspace spanned by k columns (features) and then truncating to d dimensions (`d < k, k << min(n, m)`).

# Background

Greedy algorithms (evaluating features as if they were uncorrelated, using <a href="https://satijalab.org/seurat/reference/findvariablefeatures">mean variable plot</a>), or convex relaxations (<a href="https://rdrr.io/cran/elasticnet/man/spca.html">ℓ<sub>1</sub> + ℓ<sub>2</sub> "elastic net" constraint</a>) may not help us produce a PCA plot where the PC0 and PC1 axes best explain the data. More qualitative findings will be forthcoming.

# Rank-One Updates

We have a combinatorial optimization problem, where there are `n choose k` subsets of features to be evaluated, and solutions (subsets) which differ by replacing one feature are analytically related to each other (top d singular values after rank-one update). The rank-one update library implements "append column". In a breadth-first search, first we would use a higher-quality linear algebra library to calculate SVD (singular values and right singular vectors) on a `k-1` subset (which should have many neighbors to be considered). Then, we would gather each feature that the neighboring solutions differ by.

## Updating the Covariance Matrix

We implement SVD append column, but this is computed indirectly, for right singular vectors only, as a rank-one update to the Gram matrix (covariance matrix).

## Eigenvalue perturbation

We currently don't have a good-quality guess on where the perturbed eigenvalue in each search interval is located. If our `k` covariance matrix has one row/column zeroed out, then the new eigenvalues are strictly found in open intervals between the old eigenvalues and theoretical maximum (λ' ≠ 0, λ' ≠ λ, λ ≠ tr Σ'). Jacobi's diagonalization could give us the throughput and moderate accuracy that we are looking for, and if we find a lower-fidelity Jacobi's approach, then we could greatly narrow these search intervals and initialize the analytic eigenvalue search reliably.

## Bunch's function of the eigenvalues

Bunch produced an analytic approach to the eigenvalues of a sum (a previously solved symmetric matrix and a symmetric rank-one update). We want to find the zeros of a rational function, which is the sum of `k` quotients of the form `a/(b - λ)`. As we search for a lambda which is a zero, the Taylor series of this sum is trivial to compute. Lambda guesses are applied in a batch of `k` eigenvalue guesses. A good fit to the Taylor series (beyond simply Newton's method) converges quadratically, and could reach machine precision faster than a built-in `eigh` function. However, we are actually searching for even higher throughput, and an X% relative error tolerance (e.g. 0.01%).

## Inverse iteration

We can compute eigenvalues to any desired precision, and are going to solve for one eigenvector given its eigenvalue (inverse iteration). For inverse iteration, we are going to keep our Σ' update in sparse form (LaTeX notes for our sparse update to be created soon). We will use the perturbed singular value matrix used by Ross 2008, and will take the Gram matrix (the values are located on one dense row, one dense column, and the diagonal). The singular S' (write as L) is no longer diagonal, but is very sparse.

Our inverse iteration formula is: L * L.T * (b<sub>k + 1</sub> * norm<sub>k + 1</sub>) - λ (b<sub>k + 1</sub> * norm<sub>k + 1</sub>) = b<sub>k</sub>. Our next step is to solve for a sparse (dense row, dense column, diagonal) matrix on the left, and multiple the original sparse (dense row, diagonal) matrix on the right:

(L.T - inv(L) λ) (b<sub>k + 1</sub> * norm<sub>k + 1</sub>) = inv(L) b<sub>k</sub>

The matrix on the left-hand side can be upper diagonalized.

Always, after applying the inverse matrix, b<sub>k + 1</sub> is normed by its ℓ<sub>2</sub> norm.

We require reasonable source data (no eigenvalues are near-duplicate of each other), and then a single inverse iteration can produce an eigenvector within 1% of the actual eigenvector.

# Stochastic Feature Selection

Some papers assume drawing each feature from a uniform distribution for stochastic search (subspace pursuit), and the appropriate feature removal/weighting to produce a smaller problem, at a stochastic level, still needs to be studied. After we have some datasets with a useful exact solution, then we will start to study very fast stochastic heuristics.

# References

There are some references by author name in the source code, and cited in some of the Jupyter notebooks. Full bibliography will be produced here soon.