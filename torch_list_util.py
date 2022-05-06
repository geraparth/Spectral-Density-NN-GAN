import torch
from typing import List


def tensor_list_to_vector(tor_list: List[torch.tensor]) -> torch.tensor:
    """
    Used to convert a model's parameters, which are a list of torch variables
    into a single vector (Input for Lanczos tridiagonalization)

    Input:
        Torch tensor list: List of torch tensors to convert into a single vector

    Returns:
        A torch tensor (1-D vector) created by concatening and reshaping the original list of tensors
        The shape of the vector is [w_dim, 1] where w_dim is the sum of the number of scalars in the list of tensors
    """
    return torch.cat([x.view(-1, 1) for x in tor_list], axis=0)
