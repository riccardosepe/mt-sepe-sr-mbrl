from typing import Tuple

import torch
import torch.nn.functional as F

_STRAINS = ['b', 's', 'a', 'bv', 'sv', 'av']


def build_lower_triangular(array: torch.Tensor, eps: float = 0.0):
    """
    This function takes as input an array of size m=n(n+1)/2 and builds a nxn lower triangular matrix with these values
    Args:
        array: the flattened lower triangular matrix
        eps: an optional parameter, which can be summed to the diagonal in order to enforce positive-definiteness

    Returns: the lower triangular matrix
    """
    assert len(array.shape) == 1
    m = len(array)
    n = int(((1 + 8*m)**.5 - 1) / 2)
    # first n elements of array go on the diagonal
    diagonal, rest = torch.split(array, [n, m-n])
    diagonal = F.softplus(diagonal+eps)
    mat = torch.diag(diagonal+2*eps)
    # -1 because I already have the main diagonal
    indices = torch.tril_indices(n, n, -1)
    mat[indices[0], indices[1]] = rest
    return mat


def trig_transform_q(q, env_name):
    """
    This function takes an input tensor of positions (angles of joints) and returns its trig transformation, i.e. the
    sine and cosine of each angle. NB: It's only with positions, without velocities
    Args:
        q: the positions (angles)
        env_name: the name of the environment, used to retrieve state dimensions

    Returns: the tensor of transformed angles
    """
    if env_name == "pendulum":
        return torch.column_stack((torch.cos(q[:, 0]), torch.sin(q[:, 0])))

    elif env_name == "reacher" or env_name == "acrobot":
        return torch.column_stack((torch.cos(q[:, 0]), torch.sin(q[:, 0]), torch.cos(q[:, 1]), torch.sin(q[:, 1])))

    elif env_name == "cartpole":
        return torch.column_stack((q[:, 0],
                                   torch.cos(q[:, 1]), torch.sin(q[:, 1])))

    elif env_name == "cart2pole":
        return torch.column_stack((q[:, 0],
                                   torch.cos(q[:, 1]), torch.sin(q[:, 1]),
                                   torch.cos(q[:, 2]), torch.sin(q[:, 2])))

    elif env_name == "cart3pole":
        return torch.column_stack((q[:, 0],
                                   torch.cos(q[:, 1]), torch.sin(q[:, 1]),
                                   torch.cos(q[:, 2]), torch.sin(q[:, 2]),
                                   torch.cos(q[:, 3]), torch.sin(q[:, 3])))

    elif env_name == "acro3bot":
        return torch.column_stack((torch.cos(q[:, 0]), torch.sin(q[:, 0]),
                                   torch.cos(q[:, 1]), torch.sin(q[:, 1]),
                                   torch.cos(q[:, 2]), torch.sin(q[:, 2])))
    elif env_name == "jax_pendulum":
        s = []
        n = q.shape[1]
        for i in range(n):
            s.append(torch.cos(q[:, i]))
            s.append(torch.sin(q[:, i]))

        return torch.column_stack(s)
    elif env_name == "planar_pcs" or env_name == "planar_hsa":
        return q
    else:
        raise NotImplementedError(f"{env_name} not implemented in the model.")


def inverse_trig_transform_model(x, env_name):
    if env_name == "pendulum":
        return torch.cat((torch.atan2(x[:, 1], x[:, 0]).unsqueeze(1), x[:, 2:]), 1)

    elif env_name == "reacher" or env_name == "acrobot":
        return torch.cat(
            (torch.atan2(x[:, 1], x[:, 0]).unsqueeze(1), torch.atan2(x[:, 3], x[:, 2]).unsqueeze(1), x[:, 4:]), 1)

    elif env_name == "cartpole":
        return torch.cat((x[:, 0].unsqueeze(1), torch.atan2(x[:, 2], x[:, 1]).unsqueeze(1), x[:, 3:]), 1)

    elif env_name == "cart2pole":
        return torch.cat((x[:, 0].unsqueeze(1), torch.atan2(x[:, 2], x[:, 1]).unsqueeze(1),
                          torch.atan2(x[:, 4], x[:, 3]).unsqueeze(1), x[:, 5:]), 1)

    elif env_name == "cart3pole":
        return torch.cat((x[:, 0].unsqueeze(1), torch.atan2(x[:, 2], x[:, 1]).unsqueeze(1),
                          torch.atan2(x[:, 4], x[:, 3]).unsqueeze(1),
                          torch.atan2(x[:, 6], x[:, 5]).unsqueeze(1), x[:, 7:]), 1)

    elif env_name == "acro3bot":
        return torch.cat((torch.atan2(x[:, 1], x[:, 0]).unsqueeze(1), torch.atan2(x[:, 3], x[:, 2]).unsqueeze(1),
                          torch.atan2(x[:, 5], x[:, 4]).unsqueeze(1),
                          x[:, 6:]), 1)
    elif env_name == "jax_pendulum":
        s = []
        n = x.shape[1]
        assert n % 3 == 0, "Error in dimensions"
        n = n // 3
        for i in range(0, 2*n, 2):
            s.append(torch.atan2(x[:, i+1], x[:, i]).unsqueeze(1))
        s.extend([x[:, 2*n:]])
        return torch.cat(s, dim=1)
    elif env_name == "planar_pcs" or env_name == "planar_hsa":
        return x
    else:
        raise NotImplementedError(f"{env_name} not implemented in the model.")


def divide_by_strain(x: torch.Tensor, strains: tuple, num_segments: int) -> tuple:
    assert len(x.shape) == 2, "Error in tensor dimensions"
    assert len(strains) == 3, "Strains must be a 3-bit tuple"
    assert 0 < num_segments <= 2
    strain_tensors = []
    num_strains = strains.count(1)

    assert x.shape[1] == num_strains * num_segments

    folded_x = torch.stack(x.split(num_strains, dim=1))

    j = 0
    for i in range(3):
        if strains[i]:
            strain_tensors.append(folded_x[:, :, j].T)
            j += 1
        else:
            strain_tensors.append(None)

    return tuple(strain_tensors)


def get_strain_indices(num_segments: int, requested_str: str, strains: tuple) -> list:
    """
    This function returns the indices of the passed strain inside a state vector s=(q,qdot) for a robot with n segments
    taking into account which strains are active
    Args:
        num_segments: the number of segments
        requested_str: the strain one wants to receive. It's a string in ['b', 's', 'a', 'bv', 'sv', 'av']
        strains: which strains are currently activated (as zeros and ones)

    Returns: A tuple which contains the requested indices
    """
    assert len(
        strains) == 3, "The strains should always be a tuple of bits of length 3"
    assert requested_str in _STRAINS, f"The requested strain must be a string in {_STRAINS}"
    assert 0 < num_segments <= 2, "The segments can only be 1 or 2"

    requested = _STRAINS.index(requested_str)

    if not bool(strains[requested % 3]):
        raise ValueError(
            "The requested strain is not active in the current configuration.")

    n = 0
    for i, a in enumerate(reversed(strains)):
        n += 2**i * a

    idx = []
    j = 0
    for a in strains:
        if bool(a):
            idx.append(j)
            j += 1
        else:
            idx.append(None)

    n_active = strains.count(1)

    s = (n_active * num_segments * (requested // 3)) + idx[requested % 3]
    t = n_active * num_segments * ((requested // 3) + 1)

    return list(range(s, t, n_active))
