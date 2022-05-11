import time
from typing import Any, Callable, Text, Tuple
import warnings

from torch.autograd import Variable

import matrix_vector_product
import numpy as np
import torch


def lanczos_algorithm(mvp_fn: Callable[[torch.tensor], torch.tensor],
                      dim: int,
                      order: int,
                      random_seed: int = 0,
                      only_gpu: bool = True) -> Tuple[torch.tensor, torch.tensor]:
    #     device_selector = DeviceSelector(only_gpu)

    #     with tf.device(device_selector.default):

    float_dtype = torch.float64
    tridiag = Variable(torch.zeros((order, order), dtype=float_dtype))
    vecs = Variable(torch.zeros((dim, order), dtype=float_dtype))
    torch.manual_seed(random_seed)
    init_vec = (-2) * torch.rand(dim, 1, dtype=float_dtype) + 1  # between -1 and 1 (r1 - r2) * torch.rand(a, b) + r2
    init_vec = init_vec / torch.linalg.norm(init_vec)
    vecs[:, 0:1] = init_vec  # tensor.assign was used earlier, check this
    beta = 0
    v_old = torch.zeros((dim, 1), dtype=float_dtype)

    for i in range(order):
        ts = time.time()
        v = vecs[:, i:i + 1]
        with tf.device(device_selector.accelerator):  # check this
            tss = time.time()
            w = mvp_fn(v.type(torch.FloatTensor))
            w = w.type(float_dtype)
            time_mvp = time.time() - tss

        w = w - beta * v_old
        alpha = torch.matmul(w.T, v)
        tridiag[i:i + 1, i:i + 1] = alpha  # tensor.assign was used earlier, check this
        w = w - alpha * v
        for j in range(i):
            tau = vecs[:, j:j + 1]
            coeff = torch.matmul(w.T, tau)
            w = w - coeff * tau

        beta = torch.linalg.norm(w)
        if beta < 1e-6:
            warning_msg = ("Possible numerical stability issues in Lanczos: "
                           "got beta = {} in iteration {}".format(beta.numpy(), i))
            warnings.warn(warning_msg)

        if i + 1 < order:
            tridiag[i, i + 1] = beta
            tridiag[i + 1, i] = beta
            vecs[:, i + 1:i + 2] = (w / beta)

        v_old = v

        info = "Iteration {}/{} done in {:.2f}s (MVP: {:.2f}s).".format(
            i, order, time.time() - ts, time_mvp)
        print(info)

    return vecs, tridiag


def approximate_hessian(model,  #: tf.keras.Model,
                        loss_function: Callable[[tf.keras.Model, Any],
                                                torch.tensor],
                        dataset,  #: tf.data.Dataset,
                        order: int,
                        reduce_op: Text = "MEAN",
                        random_seed: int = 0,
                        only_gpu: bool = True) -> Tuple[torch.tensor, torch.tensor]:

    def hessian_vector_product(v: torch.tensor):
        return matrix_vector_product.model_hessian_vector_product(
            loss_function, model, dataset, v, reduce_op=reduce_op)

    w_dim = sum((np.prod(w.shape) for w in list(model.parameters())))

    return lanczos_algorithm(
        hessian_vector_product, w_dim, order, random_seed=random_seed,
        only_gpu=only_gpu)
