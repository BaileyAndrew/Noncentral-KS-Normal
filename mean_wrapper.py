import numpy as np
from GmGM import Dataset
from GmGM.typing import Axis
from typing import Callable

class NoncentralKS:
    """
    Used to estimate the mean and precisions of a noncentral
    Kronecker-sum-structured distribution.

    Wraps a central Kronecker-sum-structured distribution parameter estimator

    Note that the implementation could be improved to use rank-one updates of
    the sufficient statistics (Gram matrices), and even could update the
    eigenvectors directly if this code were made more tightly intertwined with
    a central KS estimator.  However, this implementation is simpler and can
    be used with any estimator, hence our preference for implementing our
    paper's experiments with this version.

    Parameters:
    -----------
    estimator : callable[Dataset -> dict[Axis, np.ndarray]]
        An object that implements the `fit` method, which takes in a data matrix
        and returns the estimated mean and precision of the distribution.

        Output is dictionary of precision matrices
    initial_mean : tuple[dict[Axis, np.ndarray], float]
        The initial mean estimate of the distribution.
    initial_precision : dict[Axis, np.ndarray]
        The initial precision estimate of the distribution.
    """

    def __init__(
        self,
        estimator: Callable[[Dataset], dict[Axis, np.ndarray]],
        initial_mean: tuple[dict[Axis, np.ndarray], float],
        initial_precision: dict[Axis, np.ndarray],
    ) -> None:
        self.estimator = estimator
        self.column_means = initial_mean[0]
        self.full_mean = initial_mean[1]
        self.precision = initial_precision

    def fit(self, data: Dataset) -> tuple[
        tuple[dict[Axis, np.ndarray], float],
        dict[Axis, np.ndarray]
    ]:
        """
        Estimate the mean and precision of the distribution.

        Parameters:
        -----------
        data : Dataset
            The data matrix to estimate the distribution from.

        Returns:
        --------
        dict[Axis, np.ndarray]
            The estimated mean and precision of the distribution.
        """
        
        if len(data.dataset) != 1:
            raise ValueError("NoncentralKS only supports one dataset at a time")
        key = list(data.dataset.keys())[0]

        orig_data = data.dataset[key]
        indices = {
            ell: data.structure[key].index(ell) for ell in data.structure[key]
        }

        # Remove the mean estimate from the data
        data.dataset = {key: orig_data - self.full_mean}
        for axis in data.all_axes:
            data.dataset = {
                key: data.dataset[key] - self.column_means[axis].reshape(
                    *(
                        [1] * indices[axis]
                        + [-1]
                        + [1] * (orig_data.ndim - indices[axis] - 1)
                    )
                )
            }

        # Update precision estimates
        self.precision = self.estimator(data)

        # Update the mean estimates
        self.column_means, self.full_mean = mean_estimator(
            orig_data,
            self.precision,
            (self.column_means, self.full_mean),
            np.prod(orig_data.shape),
            data.structure[key],
        )

        return (self.column_means, self.full_mean), self.precision
    
def vec_kron_sum(Xs: list) -> np.array:
    """Compute the Kronecker vector-sum"""
    if len(Xs) == 1:
        return Xs[0]
    elif len(Xs) == 2:
        return np.kron(Xs[0], np.ones(Xs[1].shape[0])) + np.kron(np.ones(Xs[0].shape[0]), Xs[1])
    else:
        d_slash0 = np.prod([X.shape[0] for X in Xs[1:]])
        return (
            np.kron(Xs[0], np.ones(d_slash0))
            + np.kron(np.ones(Xs[0].shape[0]), vec_kron_sum(Xs[1:]))
        )

def mean_estimator(
    data: np.ndarray,
    Psis: dict[Axis, np.ndarray],
    initial_mean: tuple[dict[Axis, np.ndarray], float],
    d_full: float,
    axes: list[Axis],
) -> tuple[dict[Axis, np.ndarray], float]:
    # Derived parameters for our mean problem
    means = initial_mean[0]
    full_mean = initial_mean[1]
    lsum_Psis = {ell: Psis[ell].sum(axis=1) for ell in axes}
    sum_Psis = {ell: lsum_Psis[ell].sum() for ell in axes}
    ds = {ell: Psis[ell].shape[0] for ell in axes}
    d_slashes = {ell: d_full / ds[ell] for ell in axes}
    sum_Psis_slashes = {
        ell_prime: sum([
            d_slashes[ell] / ds[ell_prime] * sum_Psis[ell]
            for ell in axes if ell != ell_prime
        ])
        for ell_prime in axes
    }
    indices_dict = {ell: axes.index(ell) for ell in axes}

    # The matrix that needs to be inverted
    A = {
        ell: (
            d_slashes[ell] * Psis[ell]
            + sum_Psis_slashes[ell] * np.eye(ds[ell])
        )
        for ell in axes
    }
    A_inv = {ell: np.linalg.pinv(A[ell]) for ell in axes}

    # The data contribution
    def datatrans(ell, data, Psis):
        # Sum along all axes but ell
        base = data.sum(axis=tuple([
            indices_dict[ell_prime] for ell_prime in axes
            if ell_prime != ell
        ]))
        base = Psis[ell] @ base

        for ell_prime in axes:
            if ell_prime == ell:
                continue
            # Sum along all axes but ell and ell_prime
            to_add = data.sum(axis=tuple([
                indices_dict[_ell] for _ell in axes
                if _ell != ell and _ell != ell_prime
            ]))
            
            # Multiply by Psi_{ell_prime} and then sum along ell_prime
            if indices_dict[ell_prime] < indices_dict[ell]:
                to_add = (lsum_Psis[ell_prime] @ to_add)
            else:
                to_add = (lsum_Psis[ell_prime] @ to_add.T)

            base += to_add

        return base

    b_bases = {
        ell: datatrans(ell, data, Psis)
        for ell in axes
    }
    max_cycles = 15
    for cycle in range(max_cycles):
        for ell in axes:
            # Preliminary calculations
            mean_lsum = (
                vec_kron_sum([
                    means[ell_prime]
                    for ell_prime in axes
                    if ell != ell_prime
                ])
                @ vec_kron_sum([
                    lsum_Psis[ell_prime]
                    for ell_prime in axes
                    if ell != ell_prime
                ])
            )

            b = (
                d_slashes[ell] * full_mean * lsum_Psis[ell]
                + full_mean * sum_Psis[ell]
                + mean_lsum
                - b_bases[ell]
            )
            A_inv_b = A_inv[ell] @ b
            means[ell] = (A_inv_b.sum() / A_inv[ell].sum()) * A_inv[ell].sum(axis=0) - A_inv_b
            
        full_mean = (
            (data.reshape(-1) - vec_kron_sum(list(means.values())))
            @ vec_kron_sum(list(lsum_Psis.values()))
            / sum(d_slashes[ell] * sum_Psis[ell] for ell in axes)
        )
    return means, full_mean