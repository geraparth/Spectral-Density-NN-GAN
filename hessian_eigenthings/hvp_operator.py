"""
This module defines a linear operator to compute the hessian-vector product
for a given pytorch model using subsampled data.
"""

from typing import Callable


import torch
import torch.nn as nn
import torch.utils.data as data


import hessian_eigenthings.utils as utils

from hessian_eigenthings.operator import Operator


class HVPOperator(Operator):
    """
    Use PyTorch autograd for Hessian Vec product calculation
    model:  PyTorch network to compute hessian for
    dataloader: pytorch dataloader that we get examples from to compute grads
    loss:   Loss function to descend (e.g. F.cross_entropy)
    use_gpu: use cuda or not
    max_possible_gpu_samples: max number of examples per batch using all GPUs.
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: data.DataLoader,
        criterion: Callable[[torch.Tensor], torch.Tensor],
        use_gpu: bool = True,
        fp16: bool = False,
        full_dataset: bool = True,
        max_possible_gpu_samples: int = 256,
    ):
        size = int(sum(p.numel() for p in model.parameters()))
        super().__init__(size)
        self.grad_vec = torch.zeros(size)
        self.model = model
        if use_gpu:
            self.model = self.model.cuda()
        self.dataloader = dataloader
        # Make a copy since we will go over it a bunch
        self.dataloader_iter = iter(dataloader)
        self.criterion = criterion
        self.use_gpu = use_gpu
        self.fp16 = fp16
        self.full_dataset = full_dataset
        self.max_possible_gpu_samples = max_possible_gpu_samples

        if not hasattr(self.dataloader, '__len__') and self.full_dataset:
            raise ValueError("For full-dataset averaging, dataloader must have '__len__'")

    def apply(self, vec: torch.Tensor):
        """
        Returns H*vec where H is the hessian of the loss w.r.t.
        the vectorized model parameters
        """
        if self.full_dataset:
            return self._apply_full(vec)
        else:
            return self._apply_batch(vec)

    def _apply_batch(self, vec: torch.Tensor) -> torch.Tensor:
        """
        Computes the Hessian-vector product for a mini-batch from the dataset.
        """
        # compute original gradient, tracking computation graph
        self._zero_grad()
        grad_vec = self._prepare_grad()
        self._zero_grad()
        # take the second gradient
        # this is the derivative of <grad_vec, v> where <,> is an inner product.
        hessian_vec_prod_dict = torch.autograd.grad(
            grad_vec, self.model.parameters(), grad_outputs=vec, only_inputs=True
        )
        # concatenate the results over the different components of the network
        hessian_vec_prod = torch.cat([g.contiguous().view(-1) for g in hessian_vec_prod_dict])
        hessian_vec_prod = utils.maybe_fp16(hessian_vec_prod, self.fp16)
        return hessian_vec_prod

    def _apply_full(self, vec: torch.Tensor) -> torch.Tensor:
        """
        Computes the Hessian-vector product averaged over all batches in the dataset.

        """
        n = len(self.dataloader)
        hessian_vec_prod = None
        for _ in range(n):
            if hessian_vec_prod is not None:
                hessian_vec_prod += self._apply_batch(vec)
            else:
                hessian_vec_prod = self._apply_batch(vec)
        hessian_vec_prod = hessian_vec_prod / n
        return hessian_vec_prod

    def _zero_grad(self):
        """
        Zeros out the gradient info for each parameter in the model
        """
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.zero_()

    def _prepare_grad(self) -> torch.Tensor:
        """
        Compute gradient w.r.t loss over all parameters and vectorize
        """
        try:
            all_inputs, all_targets = next(self.dataloader_iter)
        except StopIteration:
            self.dataloader_iter = iter(self.dataloader)
            all_inputs, all_targets = next(self.dataloader_iter)

        num_chunks = max(1, len(all_inputs) // self.max_possible_gpu_samples)

        grad_vec = None

        # This will do the "gradient chunking trick" to create micro-batches
        # when the batch size is larger than what will fit in memory.
        # WARNING: this may interact poorly with batch normalization.

        input_microbatches = all_inputs.chunk(num_chunks)
        target_microbatches = all_targets.chunk(num_chunks)
        for input, target in zip(input_microbatches, target_microbatches):
            if self.use_gpu:
                input = input.cuda()
                target = target.cuda()

            output = self.model(input)
            loss = self.criterion(output, target)
            grad_dict = torch.autograd.grad(
                loss, self.model.parameters(), create_graph=True
            )
            if grad_vec is not None:
                grad_vec += torch.cat([g.contiguous().view(-1) for g in grad_dict])
            else:
                grad_vec = torch.cat([g.contiguous().view(-1) for g in grad_dict])
            grad_vec = utils.maybe_fp16(grad_vec, self.fp16)
        grad_vec /= num_chunks
        self.grad_vec = grad_vec
        return self.grad_vec
    
    
class HVPOperatorGAN(Operator):
    """
    Use PyTorch autograd for Hessian Vec product calculation
    model:  PyTorch network to compute hessian for
    dataloader: pytorch dataloader that we get examples from to compute grads
    loss:   Loss function to descend (e.g. F.cross_entropy)
    use_gpu: use cuda or not
    max_possible_gpu_samples: max number of examples per batch using all GPUs.
    """

    def __init__(
        self,
        gen: nn.Module,
        dis: nn.Module,
        dataloader: data.DataLoader,
        g_loss: Callable[[torch.Tensor], torch.Tensor],
        d_loss: Callable[[torch.Tensor], torch.Tensor],
        input_dim: int,
        use_gpu: bool = True,
        fp16: bool = False,
        full_dataset: bool = False,
        max_possible_gpu_samples: int = 256,
        model: str='gen'
    ):
        
        if model == 'gen':
            size = int(sum(p.numel() for p in gen.parameters()))
        else:
            size = int(sum(p.numel() for p in dis.parameters()))
        print(size)
        super().__init__(size)
        self.model = model
        self.gen = gen
        self.dis = dis
        self.grad_vec = torch.zeros(size)
        self.dataloader = dataloader
        self.input_dim = input_dim
        
        # Make a copy since we will go over it a bunch
        self.dataloader_iter = iter(dataloader)
        self.g_criterion = g_loss
        self.d_criterion = d_loss
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.gen = self.gen.cuda()
            self.dis = self.dis.cuda()
        self.fp16 = fp16
        self.full_dataset = full_dataset
        self.max_possible_gpu_samples = max_possible_gpu_samples

        if not hasattr(self.dataloader, '__len__') and self.full_dataset:
            raise ValueError("For full-dataset averaging, dataloader must have '__len__'")

    def apply(self, vec: torch.Tensor):
        """
        Returns H*vec where H is the hessian of the loss w.r.t.
        the vectorized model parameters
        """
        if self.full_dataset:
            return self._apply_full(vec)
        else:
            return self._apply_batch(vec)

    def _apply_batch(self, vec: torch.Tensor) -> torch.Tensor:
        """
        Computes the Hessian-vector product for a mini-batch from the dataset.
        """
        # compute original gradient, tracking computation graph
        self._zero_grad()
        grad_vec = self._prepare_grad()
        self._zero_grad()
        # take the second gradient
        # this is the derivative of <grad_vec, v> where <,> is an inner product.
        if self.model == 'gen':
            hessian_vec_prod_dict = torch.autograd.grad(
                grad_vec, self.gen.parameters(), grad_outputs=vec, only_inputs=True
            )
        else:
            hessian_vec_prod_dict = torch.autograd.grad(
                grad_vec, self.dis.parameters(), grad_outputs=vec, only_inputs=True
            )
        # concatenate the results over the different components of the network
        hessian_vec_prod = torch.cat([g.contiguous().view(-1) for g in hessian_vec_prod_dict])
        hessian_vec_prod = utils.maybe_fp16(hessian_vec_prod, self.fp16)
        return hessian_vec_prod

    def _apply_full(self, vec: torch.Tensor) -> torch.Tensor:
        """
        Computes the Hessian-vector product averaged over all batches in the dataset.

        """
        n = len(self.dataloader)
        hessian_vec_prod = None
        for _ in range(n):
            if hessian_vec_prod is not None:
                hessian_vec_prod += self._apply_batch(vec)
            else:
                hessian_vec_prod = self._apply_batch(vec)
        hessian_vec_prod = hessian_vec_prod / n
        return hessian_vec_prod

    def _zero_grad(self):
        """
        Zeros out the gradient info for each parameter in the model
        """
        if self.model == 'gen':
            for p in self.gen.parameters():
                if p.grad is not None:
                    p.grad.data.zero_()
        else:
            for p in self.dis.parameters():
                if p.grad is not None:
                    p.grad.data.zero_()

    def _prepare_grad(self) -> torch.Tensor:
        """
        Compute gradient w.r.t loss over all parameters and vectorize
        """
        try:
            all_inputs = next(self.dataloader_iter)[0]
        except StopIteration:
            self.dataloader_iter = iter(self.dataloader)
            all_inputs = next(self.dataloader_iter)[0]
            
        num_chunks = max(1, len(all_inputs) // self.max_possible_gpu_samples)

        grad_vec = None

        # This will do the "gradient chunking trick" to create micro-batches
        # when the batch size is larger than what will fit in memory.
        # WARNING: this may interact poorly with batch normalization.

        input_microbatches = all_inputs.chunk(num_chunks)
#         target_microbatches = all_targets.chunk(num_chunks)
#         print('Total chunks:', num_chunks)
        for i, x in enumerate(input_microbatches):
            if self.use_gpu:
                x = x.cuda()
#                 target = target.cuda()
            
            if self.model == 'gen':
                z = torch.randn(x.shape[0], self.input_dim).to(x)
                x_hat = self.gen(z)
                y_hat = self.dis(x_hat)
                y = self.dis(x)
                loss = self.g_criterion(y_hat, y)
                grad_dict = torch.autograd.grad(
                    loss, self.gen.parameters(), create_graph=True
                )
            else:
                z = torch.randn(x.shape[0], self.input_dim).to(x)
                x_hat = self.gen(z).detach()
                y_hat = self.dis(x_hat)
                y = self.dis(x)
                loss = self.d_criterion(y_hat, y)
                grad_dict = torch.autograd.grad(
                    loss, self.dis.parameters(), create_graph=True
                )
            if grad_vec is not None:
                grad_vec += torch.cat([g.contiguous().view(-1) for g in grad_dict])
            else:
                grad_vec = torch.cat([g.contiguous().view(-1) for g in grad_dict])
            grad_vec = utils.maybe_fp16(grad_vec, self.fp16)
            
#             if i % 10 == 0:
#                 print(f'Chunk: {i}/{num_chunks}')
        grad_vec /= num_chunks
        self.grad_vec = grad_vec
        return self.grad_vec
