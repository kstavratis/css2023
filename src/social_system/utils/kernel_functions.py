import numpy as np
from numpy import typing as npt


def __identity(
        opinion_values: npt.NDArray[np.float64],
        opinion_tolerances: npt.NDArray[np.float64],
) -> npt.NDArray[np.int8]:
    
    return np.ones(shape=(opinion_values.size, opinion_values.size))

def __bc(
        opinion_values: npt.NDArray[np.float64],
        opinion_tolerances: npt.NDArray[np.float64],
) -> npt.NDArray[np.int8]:
    """
    'Bounded Confidence' kernel function
    `k(o, o', t) = if |o - o'| < t => 1,  else 0`

    Parameters
    ----------
    opinion_values : npt.NDArray[np.float64]
        A 1D numpy array containing the opinion value of each agent.
    opinion_tolerances : npt.NDArray[np.float64]
        A 1D numpy array containing the tolerance value of each agent.

    Returns
    -------
    npt.NDArray[np.int8]
        A 2D numpy array containing the kernel function result
        among all agents of the system
        `output[i, j] === k(o_i, o_j, t_i)`
    """
    
    # Dynamic sanity checks.
    assert opinion_values.ndim == 1
    assert opinion_tolerances.ndim == 1
    assert opinion_values.size == opinion_tolerances.size,\
        f'`opinion_values` and `opinion_tolerances` must have the same number of elements.\
            Instead `opinion_values = {opinion_values} != {opinion_tolerances} = `opinion_tolerances`'
    
    # We wish for the matrix O := o - o_prime to be of the form
    # O_{ij} = o_i - o_j
    # i.e. a row i of the subtraction matrix O := o - o_prime
    # expresses the (signed) distance of the i-th agent
    # from all other agents (the columns j).

    # o_prime[i] = [o1, o2, ..., o_n]
    o_prime = np.tile(opinion_values, (opinion_values.size, 1))
    # o[i] = [o_i, o_i, ..., o_i]
    o = o_prime.T

    # "b" postfix stands for "broadcasted". 
    opinion_tolerance_b = np.tile(opinion_tolerances, (opinion_values.size, 1)).T

    return np.where(np.abs(o - o_prime) < opinion_tolerance_b, 1, 0)


def __gbc(
        opinion_values: npt.NDArray[np.float64],
        opinion_tolerances: npt.NDArray[np.float64]    
) -> npt.NDArray[np.float64]:
    """
    'Gaussian Bounded Confidence' kernel function
    `k(o, o', t) = exp{ -[(o - o')/t]^2 }`

    Parameters
    ----------
    opinion_values : npt.NDArray[np.float64]
        A 1D numpy array containing the opinion value of each agent.
    opinion_tolerances : npt.NDArray[np.float64]
        A 1D numpy array containing the tolerance value of each agent.

    Returns
    -------
    npt.NDArray[np.int8]
        A 2D numpy array containing the kernel function result
        among all agents of the system.
        `output[i, j] === k(o_i, o_j, t_i)`
    """

    # Dynamic sanity checks.
    assert opinion_values.ndim == 1
    assert opinion_tolerances.ndim == 1
    assert opinion_values.size == opinion_tolerances.size, \
        f'`opinion_values` and `opinion_tolerances` must have the same number of elements.\
            Instead `opinion_values = {opinion_values} != {opinion_tolerances} = `opinion_tolerances`'

    # Having the division by the tolerance happen here,
    # because it is easier here than broadcast it later.
    # This is acceptable, because
    # (x - x')/u = x/u - x'/u
    temp = opinion_values / opinion_tolerances

    o_prime = np.tile(temp, temp.size, 1, axis=1) # o_prime[i] = [o1, o2, ..., o_n]    
    o = o_prime.T # o[i] = [o_i, o_i, ..., o_i]
    o_minus_o_prime = o - o_prime # (x-x')/u
    
    return np.exp(-o_minus_o_prime**2)

def __ra(
        opinion_values: npt.NDArray[np.float64],
        opinion_tolerances: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """
    'Relative Agreement' model
    The relative agreement introduces a new assumption:
    individuals take into account the tolerance of their interlocutor,
    and interlocutors with a low tolerance (high confidence)
    tend to be more influential than those with a high tolerance.

    ` k(o, o', t, t') := "overlap between the ranges of o and o'" / t' - 1`
    lower bounded by `0`.

    NOTE: The value of the kernel is 1
    when the segment [o'-t',o'+t'] is totally included in the segment [o-t, o+t].
    Otherwise, it is lower than 1.

    Parameters
    ----------
    opinion_values : npt.NDArray[np.float64]
        A 1D numpy array containing the opinion value of each agent.
    opinion_tolerances : npt.NDArray[np.float64]
        A 1D numpy array containing the tolerance value of each agent.

    Returns
    -------
    npt.NDArray[np.float64]
        A 2D numpy array containing the kernel function result
        among all agents of the system.
        `output[i, j] = k(o_i, o_j, t_i, t_j)`
    """

    o_prime_left = np.tile(opinion_values - opinion_tolerances, (opinion_values.size, 1))
    o_prime_right = np.tile(opinion_values + opinion_tolerances, (opinion_values.size, 1))

    o_left, o_right = o_prime_left.T, o_prime_right.T

    # overlap[i, j] := overlap between tolerance ranges of agents i and j.
    overlap = np.minimum(o_right, o_prime_right) - np.maximum(o_left, o_prime_left)

    # Not precomputing `overlap/opinion_tolernaces`,
    # because a division by zero problem may arise.
    # This has the cost of doing computations twice
    return np.where(overlap > opinion_tolerances, overlap / opinion_tolerances - 1, 0)


kernel_functions_dict = {
    'identity': __identity,
    'bc': __bc,
    'gbc': __gbc,
    'ra': __ra 
}