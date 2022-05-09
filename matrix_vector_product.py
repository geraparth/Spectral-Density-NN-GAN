from typing import Callable, List, Text, Tuple, Union
import torch_list_util
import torch

Parameters = Union[torch.tensor, List[torch.tensor]]


def hessian_vector_product(function: Callable[[Parameters], torch.Tensor], parameters: Parameters, v: Parameters) -> Parameters:

    """
        Computes hessian vector product where v is an arbitrary vector and H is the Hessian
        of a function

        Input:
            function: A (twice) differentiable function that takes as input a tensor or a list of
            tensors and returns a scalar.

            parameters: The parameters with respect to which we want to compute the
              Hessian for the hessian vector product.

            v: An arbitrary vector or list of vectors of the same nested structure as
              `parameters`.
        Returns:
            A vector or list of vectors of the same nested structure as `parameters`, equal to H.v.
    """

    grad_f, = torch.autograd.grad(function(parameters), parameters, create_graph=True)
    z = grad_f.T @ v
    z.backward()

    return parameters.grad




