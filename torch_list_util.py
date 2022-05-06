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


def vector_to_tensor_list(vector: torch.Tensor, structure: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Converts a 1-D torch tensor into a list of torch tensors with n elements where n is the length of a dummy
    structure passed as input

    Input:
        vector : A 1-d torch tensor
        structure : A dummy list of torch tensors, whose structure/size needs to be replicated for the output

    Returns:
        tor_list :  A list of torch tensors created using elements from vector in the same shapes as structure

    """

    index = 0
    tor_list = []

    for element in structure:
        dim = element.shape
        elements = torch.prod(torch.tensor(dim))
        tor_list.append(vector[index:index + elements].reshape(dim))
        index += elements

    return tor_list
