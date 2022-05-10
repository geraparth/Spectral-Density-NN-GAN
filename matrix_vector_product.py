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


def reduce_function_over_dataset(function: Callable[[Tuple[torch.tensor, torch.tensor]], Parameters],
                                  dataset: torch.utils.data.Dataset,
                                  reduce_op: Text = "MEAN") -> Parameters:
    """Averages or sums f(x) over x in a dataset, for any arbitrary function f.

      Input:
        function: A function that take as input examples sampled from the dataset,
          and return a Tensor or list of Tensors.
        dataset: A dataset that yield the inputs to `function` over which the
          outputs of `function` should be averaged or summed.
        reduce_op: Whether to average over the dataset (if set to `MEAN`) or
          to simply sum the output tensors (if set to `SUM`).
      Returns:
        Output of `function` averaged or summed over the dataset.
      """

    assert reduce_op in ["MEAN", "SUM"]
    dataset = iter(dataset)

    # We loose a bit of generality by assuming that the dataset yield tuple of
    # tensors instead of anything that the function can take as input, only to
    # be able to get the batch size. Fine for now, maybe change later if this ever
    # becomes a restriction.

    x, y = next(dataset)
    acc = function((x, y))
    acc = [acc] if not isinstance(acc, list) else acc
    accumulated_obs = x.shape[0]
    for x, y in dataset:

        new_val = function((x, y))
        new_obs = x.shape[0]
        w_old = accumulated_obs / (accumulated_obs + new_obs)
        w_new = new_obs / (accumulated_obs + new_obs)
        new_val = [new_val] if not isinstance(new_val, list) else new_val
        for i, value in enumerate(new_val):
            if reduce_op == "SUM":
                acc[i] = acc[i] + value
            else:
                acc[i] = w_old * acc[i] + w_new * value
        accumulated_obs += new_obs
    return acc


def model_hessian_vector_product(
        loss_function,  #: Callable[[tf.keras.Model, Tuple[torch.tensor, torch.tensor]], torch.tensor],
        model,  #: tf.keras.Model,
        dataset,  #: tf.data.Dataset,
        v: torch.tensor,
        reduce_op: Text = "MEAN") -> torch.tensor:
    if reduce_op not in ["MEAN", "SUM"]:
        raise ValueError(
            "`reduce_op` must be in 'MEAN' or 'SUM', but got {}".format(reduce_op))
    v = torch_list_util.vector_to_tensor_list(v, model.trainable_variables)

    def loss_hessian_vector_product(inputs):
        return hessian_vector_product(
            lambda _: loss_function(model, inputs),
            model.trainable_variables,
            v)

    mvp_as_list_of_tensors = reduce_function_over_dataset(
        loss_hessian_vector_product,
        dataset,
        reduce_op=reduce_op)
    return torch_list_util.tensor_list_to_vector(mvp_as_list_of_tensors)





