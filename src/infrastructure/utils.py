import collections
import functools
import gc
import hashlib
import inspect
import json
import math
import os
import sys
import time
from pathlib import Path
from argparse import Namespace
from collections import OrderedDict
from functools import wraps
from matplotlib import transforms
from matplotlib.patches import Ellipse
from tqdm import tqdm
from types import MappingProxyType
from typing import *

import einops
import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import torchdiffeq
from dimarray import DimArray, Dataset
from tensordict import TensorDict
from torch.utils._pytree import tree_flatten, tree_unflatten

from infrastructure.settings import DEVICE, SEED


ModelPair = tuple[nn.Module, TensorDict]
TRAINING_DATASET_TYPES: List[str] = ["train", "valid"]
TESTING_DATASET_TYPE: str = "test"


_T = TypeVar("_T")
"""
System and model functions
"""

def stack_tensor_arr(tensor_arr: np.ndarray[torch.Tensor], dim: int = 0) -> Union[torch.Tensor, TensorDict]:
    tensor_list = [*tensor_arr.ravel()]
    if isinstance(t := tensor_list[0], torch.Tensor):
        result = torch.stack(tensor_list, dim=dim)
    else:
        result = TensorDict.maybe_dense_stack(tensor_list, dim=dim)
    return result.reshape((*tensor_arr.shape, *t.shape,))

def stack_module_arr(module_arr: np.ndarray[nn.Module]) -> ModelPair:
    params, buffers = torch.func.stack_module_state(module_arr.ravel().tolist())
    td = TensorDict({}, batch_size=module_arr.shape)

    def _unflatten(t: torch.Tensor, dim: int, shape: Tuple[int, ...]):
        if len(shape) == 0:
            return t.squeeze(dim=dim)
        elif len(shape) == 1:
            return t
        else:
            return t.unflatten(dim, shape)

    for k, v in params.items():
        td[(*k.split("."),)] = nn.Parameter(_unflatten(v, 0, module_arr.shape), requires_grad=v.requires_grad)
    for k, v in buffers.items():
        td[(*k.split("."),)] = _unflatten(v, 0, module_arr.shape)

    return module_arr.ravel()[0].to(DEVICE), td.to(DEVICE)

def stack_module_arr_preserve_reference(module_arr: np.ndarray[nn.Module]) -> ModelPair:
    flattened_td = TensorDict.maybe_dense_stack([
        TensorDict({
            k: v
            for k in dir(module) if isinstance((v := getattr(module, k)), torch.Tensor)
        }, batch_size=())
        for module in module_arr.ravel()
    ], dim=0)
    td = flattened_td.reshape(module_arr.shape)
    return module_arr.ravel()[0], td.to(DEVICE)

def run_module_arr(
        model_pair: ModelPair,
        args: Any,  # Note: a TensorDict is only checked for as the immediate argument and will not work inside a nested structure
        kwargs: Dict[str, Any] = MappingProxyType(dict())
) -> Any:
    if "TensorDict" in type(args).__name__:
        args = args.to_dict()

    reference_module, module_td = model_pair
    module_td = TensorDict(td_items(module_td), batch_size=module_td.shape)
    n = int(np.prod(module_td.shape))
    try:
        assert n > 1
        def vmap_run(module_d, ags):
            return torch.func.functional_call(reference_module, module_d, ags, kwargs)
        return multi_vmap(vmap_run, module_td.ndim, randomness="different")(module_td.to_dict(), args)
    except (AssertionError, RuntimeError):
        flat_args, args_spec = tree_flatten(args)
        single_flat_args_list = [
            [t.view(n, *t.shape[module_td.ndim:])[idx] for t in flat_args]
            for idx in range(n)
        ]
        single_args_list = [tree_unflatten(single_flat_args, args_spec) for single_flat_args in single_flat_args_list]

        # TODO: If this line breaks, replace `torch.func.functional_call` with `nn.utils.stateless.functional_call`
        single_out_list = [
            torch.func.functional_call(reference_module, module_td.view(n)[idx].to_dict(), single_args)
            for idx, single_args in enumerate(single_args_list)
        ]
        _, out_spec = tree_flatten(single_out_list[0])
        single_flat_out_list = [tree_flatten(single_out)[0] for single_out in single_out_list]
        flat_out = [
            torch.stack([*out_component_list], dim=0).view(*module_td.shape, *out_component_list[0].shape)
            for out_component_list in zip(*single_flat_out_list)
        ]
        return tree_unflatten(flat_out, out_spec)

def multi_vmap(func: Callable, n: int, **kwargs: Any) -> Callable:
    f = func
    for _ in range(n):
        f = torch.vmap(f, **kwargs)
    return f

def buffer_dict(td: TensorDict) -> nn.Module:
    def _buffer_dict(parent_module: nn.Module, td: TensorDict) -> nn.Module:
        for k, v in td.items(include_nested=False):
            if isinstance(v, torch.Tensor):
                parent_module.register_buffer(k, v)
            else:
                parent_module.register_module(k, _buffer_dict(nn.Module(), v))
        return parent_module
    return _buffer_dict(nn.Module(), td)

def td_items(td: TensorDict) -> Dict[str, torch.Tensor]:
    return {
        k if isinstance(k, str) else ".".join(k): v
        for k, v in td.items(include_nested=True, leaves_only=True)
    }

def parameter_td(m: nn.Module) -> TensorDict:
    result = TensorDict({}, batch_size=())
    for k, v in m.named_parameters():
        k_ = (*k.split("."),)
        result[k_[0] if len(k_) == 1 else k_] = v
    return result

def mask_dataset_with_total_sequence_length(ds: TensorDict, total_sequence_length: int) -> TensorDict:
    batch_size, sequence_length = ds.shape[-2:]
    ds["mask"] = torch.Tensor(torch.arange(batch_size * sequence_length) < total_sequence_length).view(
        sequence_length, batch_size
    ).mT.expand(ds.shape)
    return ds


"""
Computation
"""
def pow_series(M: torch.Tensor, n: int) -> torch.Tensor:
    N = M.shape[0]
    I = torch.eye(N, device=M.device)
    if n == 1:
        return I[None]
    else:
        k = int(math.ceil(math.log2(n)))
        bits = [M]
        for _ in range(k - 1):
            bits.append(bits[-1] @ bits[-1])

        result = I
        for bit in bits:
            augmented_bit = torch.cat([I, bit], dim=1)
            blocked_result = result @ augmented_bit
            result = torch.cat([blocked_result[:, :N], blocked_result[:, N:]], dim=0)
        return result.reshape(1 << k, N, N)[:n]

def batch_trace(x: torch.Tensor) -> torch.Tensor:
    return x.diagonal(dim1=-2, dim2=-1).sum(dim=-1)

def kl_div(cov1: torch.Tensor, cov2: torch.Tensor) -> torch.Tensor:
    return ((torch.det(cov2) / torch.det(cov1)).log() - cov1.shape[-1] + (torch.inverse(cov2) * cov1).sum(dim=(-2, -1))) / 2

def sqrtm(t: torch.Tensor) -> torch.Tensor:
    L, V = torch.linalg.eig(t)
    return (V @ torch.diag_embed(L ** 0.5) @ torch.inverse(V)).real

def complex(t: torch.Tensor | TensorDict) -> Union[torch.Tensor, TensorDict]:
    fn = lambda t_: t_ if torch.is_complex(t_) else torch.complex(t_, torch.zeros_like(t_))
    return fn(t) if isinstance(t, torch.Tensor) else t.apply(fn)

def ceildiv(a: int | torch.Tensor, b: int | torch.Tensor) -> int | torch.Tensor:
    q = -(-a // b)
    return q.to(torch.int32) if torch.is_tensor(q) else q

def ceil(a: int) -> int:
    return ceildiv(a, 1)

def T(t: torch.Tensor) -> torch.Tensor:
    return t.permute((*range(t.ndim - 1, -1, -1),))

def linspace(a: torch.Tensor, b: torch.Tensor, n: int) -> torch.Tensor:
    if not torch.is_tensor(a):
        a = torch.tensor(a)
    if not torch.is_tensor(b):
        b = torch.tensor(b)
    a, b = torch.broadcast_tensors(a, b)
    return multi_vmap(torch.linspace, a.ndim, in_dims=(0, 0, None,), out_dims=0)(a, b, n)

def geomspace(a: torch.Tensor, b: torch.Tensor, n: int) -> torch.Tensor:
    if not torch.is_tensor(a):
        a = torch.tensor(a)
    if not torch.is_tensor(b):
        b = torch.tensor(b)
    return torch.exp(linspace(torch.log(a), torch.log(b), n))

class _AugmentedModule(nn.Module):
    def __init__(self, t: torch.Tensor, module: nn.Module) -> None:
        nn.Module.__init__(self)
        self.time_scale = torch.diff(t, dim=-1)
        self.module = module
        self.bsz = t.shape[:-1]
    
    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        t_scale = self.time_scale[..., min(int(t.item()), self.time_scale.shape[-1] - 1)]
        vz = self.module(z[..., 0], z[..., 1:])
        return t_scale[..., None] * torch.cat((torch.ones(self.bsz + (1,)), vz,), dim=-1)

def batch_odeint(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    y: torch.Tensor,
    t: torch.Tensor,
    adjoint: bool,
    **kwargs,
) -> torch.Tensor:
    augmented_fn = _AugmentedModule(t, fn)
    _y = torch.cat((t[..., 0, None], y,), dim=-1)
    _t = torch.arange(0, t.shape[-1], dtype=torch.float)

    if adjoint:
        odeint_func = torchdiffeq.odeint_adjoint
    else:
        odeint_func = torchdiffeq.odeint

    _out = odeint_func(augmented_fn, _y, _t, **kwargs)
    out = _out[..., 1:]

    return out

# def hadamard_conjugation(
#         A: torch.Tensor,        # [B... x m x n]
#         B: torch.Tensor,        # [B... x p x q]
#         alpha: torch.Tensor,    # [B... x m x n]
#         beta: torch.Tensor,     # [B... x p x q]
#         C: torch.Tensor         # [B... x m x p]
# ) -> torch.Tensor:              # [B... x n x q]
#     coeff = 1 / (1 - alpha.conj()[..., :, None, :, None] * beta[..., None, :, None, :])             # [B... x m x p x n x q]
#     return torch.einsum("...mn, ...pq, ...mp, ...mpnq -> ...nq", A.conj(), B, complex(C), coeff,)   # [B... x n x q]

# def hadamard_conjugation_diff_order1(
#         A: torch.Tensor,        # [B... x m x n]
#         B: torch.Tensor,        # [B... x p x q]
#         alpha: torch.Tensor,    # [B... x m x n]
#         beta1: torch.Tensor,    # [B... x p x q]
#         beta2: torch.Tensor,    # [B... x p x q]
#         C: torch.Tensor         # [B... x m x p]
# ) -> torch.Tensor:              # [B... x n x q]
#     alpha_ = alpha.conj()[..., :, None, :, None]                                                    # [B... x m x 1 x n x 1]
#     _beta1, _beta2 = beta1[..., None, :, None, :], beta2[..., None, :, None, :]                     # [B... x 1 x p x 1 x q]
#     coeff = alpha_ / ((1 - alpha_ * _beta1) * (1 - alpha_ * _beta2))                                # [B... x m x p x n x q]
#     return torch.einsum("...mn, ...pq, ...mp, ...mpnq -> ...nq", A.conj(), B, complex(C), coeff,)   # [B... x n x q]

# def hadamard_conjugation_diff_order2(
#         B: torch.Tensor,        # [B... x p x q]
#         beta1: torch.Tensor,    # [B... x p x q]
#         beta2: torch.Tensor,    # [B... x p x q]
#         C: torch.Tensor         # [B... x p x p]
# ) -> torch.Tensor:              # [B... x q x q]
#     beta1_, _beta1 = beta1[..., :, None, :, None].conj(), beta1[..., None, :, None, :]  # b1_ik, b1_jl
#     beta2_, _beta2 = beta2[..., :, None, :, None].conj(), beta2[..., None, :, None, :]  # b2_ik, b2_jl

#     beta12 = beta1_ * _beta2                                                            # b1_ik * b2_jl
#     beta21 = einops.rearrange(beta12, "... i j k l -> ... j i l k").conj()              # b2_ik * b1_jl
#     beta11, beta22 = (beta1_ * _beta1), (beta2_ * _beta2),                              # b1_ik * b1_jl, b2_ik * b2_jl,

#     coeff = 1 - beta12 * beta21
#     for t in (beta11, beta12, beta21, beta22,):
#         coeff.div_(1 - t)
#     return torch.einsum("...mn, ...pq, ...mp, ...mpnq -> ...nq", B.conj(), B, C, coeff,)


def hadamard_conjugation(
        A: torch.Tensor,        # [B... x m x n]
        B: torch.Tensor,        # [B... x p x q]
        alpha: torch.Tensor,    # [B... x m x n]
        beta: torch.Tensor,     # [B... x p x q]
        C: torch.Tensor         # [B... x m x p]
) -> torch.Tensor:              # [B... x n x q]
    coeff = 1 / (1 - alpha[..., :, None, :, None] * beta[..., None, :, None, :])            # [B... x m x p x n x q]
    return torch.einsum("...mn, ...pq, ...mp, ...mpnq -> ...nq", A, B, complex(C), coeff,)  # [B... x n x q]

def hadamard_conjugation_diff_order1(
        A: torch.Tensor,        # [B... x m x n]
        B: torch.Tensor,        # [B... x p x q]
        alpha: torch.Tensor,    # [B... x m x n]
        beta1: torch.Tensor,    # [B... x p x q]
        beta2: torch.Tensor,    # [B... x p x q]
        C: torch.Tensor         # [B... x m x p]
) -> torch.Tensor:              # [B... x n x q]
    alpha_ = alpha[..., :, None, :, None]                                                   # [B... x m x 1 x n x 1]
    _beta1, _beta2 = beta1[..., None, :, None, :], beta2[..., None, :, None, :]             # [B... x 1 x p x 1 x q]
    coeff = alpha_ / ((1 - alpha_ * _beta1) * (1 - alpha_ * _beta2))                        # [B... x m x p x n x q]
    return torch.einsum("...mn, ...pq, ...mp, ...mpnq -> ...nq", A, B, complex(C), coeff,)  # [B... x n x q]

def hadamard_conjugation_diff_order2(
        B: torch.Tensor,        # [B... x p x q]
        beta1: torch.Tensor,    # [B... x p x q]
        beta2: torch.Tensor,    # [B... x p x q]
        C: torch.Tensor         # [B... x p x p]
) -> torch.Tensor:              # [B... x q x q]
    beta1_, _beta1 = beta1[..., :, None, :, None], beta1[..., None, :, None, :]             # b1_ik, b1_jl
    beta2_, _beta2 = beta2[..., :, None, :, None], beta2[..., None, :, None, :]             # b2_ik, b2_jl

    beta12 = beta1_ * _beta2                                                                # b1_ik * b2_jl
    beta21 = einops.rearrange(beta12, "... i j k l -> ... j i l k")                         # b2_ik * b1_jl
    beta11, beta22 = (beta1_ * _beta1), (beta2_ * _beta2),                                  # b1_ik * b1_jl, b2_ik * b2_jl,

    coeff = 1 - beta12 * beta21
    for t in (beta11, beta12, beta21, beta22,):
        coeff.div_(1 - t)
    return torch.einsum("...mn, ...pq, ...mp, ...mpnq -> ...nq", B, B, C, coeff,)

def inverse(A: torch.Tensor) -> torch.Tensor:
    try:
        return torch.inverse(A)
    except torch._C._LinAlgError:
        U, S, VT = torch.linalg.svd(A)
        VSinv: torch.Tensor = VT.mT / S[..., None, :]
        VSinv.nan_to_num_(nan=0.0, posinf=torch.inf, neginf=-torch.inf)
        VSinvUT = VSinv @ U.mT
        VSinvUT.nan_to_num_(nan=0.0, posinf=torch.inf, neginf=-torch.inf)
        return VSinvUT
    
def eig_some(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz = A.shape[:-2]
    L, V = torch.linalg.eig(A)
    Vinv = torch.inverse(V)
    n_nonzero = torch.sum(L != 0, dim=-1)
    max_n_nonzero = torch.max(n_nonzero).item()
    indices = torch.topk(torch.abs(L), k=max_n_nonzero, dim=-1).indices
    L = torch.gather(L, -1, indices)

    vmap_gather = multi_vmap(lambda t, idx: t[:, idx], n=len(bsz))
    V = vmap_gather(V, indices)
    Vinv = vmap_gather(Vinv.mT, indices).mT
    return V, L, Vinv

def _torch_schur(A: torch.Tensor, sort: Callable[[torch.Tensor], torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    n = A.shape[-1]

    A_complex = torch.complex(A, torch.zeros_like(A))
    L, V = torch.linalg.eig(A_complex)                          # [B... x N], [B... x N x N]
    order = torch.argsort(sort(L), dim=-1)                      # [B... x N]
    sorted_L = torch.take_along_dim(L, order, dim=-1)           # [B... x N]

    P = torch.eye(n, dtype=V.dtype)[order].mT                   # [B... x N x N]
    sorted_V = V @ P                                            # [B... x N x N]

    Q, R = torch.linalg.qr(sorted_V)                            # [B... x N x N], [B... x N x N]
    D = torch.diagonal(R, dim1=-2, dim2=-1)                     # [B... x N]
    R = R / D.unsqueeze(-2)                                     # [B... x N x N] / [B... x 1 x N]

    T = R @ torch.diag_embed(sorted_L) @ torch.inverse(R)       # [B... x N x N]
    return T.real, Q.real

def solve_discrete_are(
        A: torch.Tensor,
        B: torch.Tensor,
        Q: torch.Tensor,
        R: torch.Tensor,
        balanced: bool = False,
        use_scipy: bool = False,
        precision: torch.dtype = torch.float64,
) -> torch.Tensor:
    original_dtype = A.dtype
    A, B, Q, R = A.to(precision), B.to(precision), Q.to(precision), R.to(precision)

    with default_dtype(precision):
        bsz = A.shape[:-2]
        Q = 0.5 * (Q + Q.mT)
        R = 0.5 * (R + R.mT)
        M, N = B.shape[-2:]

        try:
            raise Exception()
            I = torch.eye(M).expand(bsz + (M, M,))
            Z = Fn.pad(A, (0, M, 0, M,), mode="constant", value=0.0) + torch.cat([
                -B @ torch.inverse(R) @ B.mT, I
            ], dim=-2) @ torch.inverse(A.mT) @ torch.cat([
                -Q, I
            ], dim=-1)

            if use_scipy:
                T, U, _ = scipy.linalg.schur(Z.numpy(force=True), sort="iuc")
                T, U = torch.tensor(T, dtype=precision), torch.tensor(U, dtype=precision)
            else:
                T, U = _torch_schur(Z, sort=torch.abs)
        except Exception:
            LEFT = torch.cat([
                torch.cat([A, torch.zeros((M, M,)), B], dim=-1),
                torch.cat([-Q, torch.eye(M), torch.zeros((M, N,))], dim=-1),
                torch.cat([torch.zeros((N, M,)), torch.zeros((N, M,)), R], dim=-1),
            ], dim=-2)
            RIGHT = torch.cat([
                torch.cat([torch.eye(M), torch.zeros((M, M,)), torch.zeros((M, N,))], dim=-1),
                torch.cat([torch.zeros((M, M,)), A.mT, torch.zeros((M, N,))], dim=-1),
                torch.cat([torch.zeros((N, M,)), -B.mT, torch.zeros((N, N,))], dim=-1),
            ], dim=-2)

            if balanced:
                # xGEBAL does not remove the diagonals before scaling. Also
                # to avoid destroying the Symplectic structure, we follow Ref.3
                ABS = torch.abs(LEFT) + torch.abs(RIGHT)
                ABS[..., range(2 * M + N), range(2 * M + N)] = 0

                sca = torch.tensor(scipy.linalg.matrix_balance(ABS.numpy(force=True), separate=1, permute=0)[1][0])
                # do we need to bother?
                if not torch.allclose(sca, torch.ones_like(sca)):
                    # Now impose diag(D,inv(D)) from Benner where D is
                    # square root of s_i/s_(n+i) for i=0,....
                    sca = torch.log2(sca)
                    # NOTE: Py3 uses "Bankers Rounding: round to the nearest even" !!
                    s = torch.round((sca[..., M:2 * M] - sca[..., :M]) / 2)
                    sca = 2 ** torch.cat([s, -s, sca[..., 2 * M:]], dim=-1)
                    # Elementwise multiplication via broadcasting.
                    elwisescale = sca[..., None] / sca
                    LEFT *= elwisescale
                    LEFT *= elwisescale

            # Deflate the pencil by the R column ala Ref.1
            q_of_qr: torch.Tensor = torch.linalg.qr(LEFT[..., -N:], mode="complete")[0]
            LEFT = q_of_qr[..., N:].conj().mT @ LEFT[..., :2 * M]
            RIGHT = q_of_qr[..., N:].conj().mT @ RIGHT[..., :2 * M]

            # U = torch.tensor(scipy.linalg.ordqz(
            #     LEFT.numpy(force=True), RIGHT.numpy(force=True),
            #     sort="iuc", output="real", overwrite_a=True, overwrite_b=True,
            # )[-1])
            def iuc(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
                return (np.abs(alpha / beta) <= 1)

            AA, BB, alpha, beta, Q, U = map(torch.tensor, scipy.linalg.ordqz(
                LEFT.numpy(force=True), RIGHT.numpy(force=True),
                sort=iuc, output="real", overwrite_a=True, overwrite_b=True,
            ))
            
        U11 = U[..., :M, :M]
        U21 = U[..., M:2 * M, :M]
        P: torch.Tensor = U21 @ torch.inverse(U11)

    return P.to(original_dtype)

def test_discrete_are(
        A: torch.Tensor,
        B: torch.Tensor,
        Q: torch.Tensor,
        R: torch.Tensor,
        P: torch.Tensor,
) -> torch.Tensor:
    P = (P + P.mT) / 2
    ATP = A.mT @ P
    ATPB = ATP @ B
    return ATP @ A - P - ATPB @ torch.inverse(R + B.mT @ P @ B) @ ATPB.mT + Q

def solve_continuous_are(
        A: torch.Tensor,
        B: torch.Tensor,
        Q: torch.Tensor,
        R: torch.Tensor,
        use_scipy: bool = False,
        precision: torch.dtype = torch.float64,
) -> torch.Tensor:
    original_dtype = A.dtype
    A, B, Q, R = A.to(precision), B.to(precision), Q.to(precision), R.to(precision)

    with default_dtype(precision):
        Q = 0.5 * (Q + Q.mT)
        R = 0.5 * (R + R.mT)
        M, N = B.shape[-2:]

        try:
            Z = torch.cat([
                torch.cat([A, -B @ torch.inverse(R) @ B.mT], dim=-1),
                torch.cat([-Q, -A.mT], dim=-1),
            ], dim=-2)

            if use_scipy:
                T, U, _ = scipy.linalg.schur(Z.numpy(force=True), sort="lhp")
                T, U = torch.tensor(T, dtype=precision), torch.tensor(U, dtype=precision)
            else:
                T, U = _torch_schur(Z, sort=torch.real)
        except Exception:
            LEFT = torch.cat([
                torch.cat([A, torch.zeros((M, M,)), B], dim=-1),
                torch.cat([-Q, -A.mT, torch.zeros((M, N,))], dim=-1),
                torch.cat([torch.zeros((N, M,)), B.mT, R], dim=-1),
            ], dim=-2)
            RIGHT = torch.cat([
                torch.cat([torch.eye(M), torch.zeros((M, M,)), torch.zeros((M, N,))], dim=-1),
                torch.cat([torch.zeros((M, M,)), torch.eye(M), torch.zeros((M, N,))], dim=-1),
                torch.cat([torch.zeros((N, M,)), torch.zeros((N, M,)), torch.zeros((N, N,))], dim=-1),
            ], dim=-2)

            # Deflate the pencil to 2m x 2m ala Ref.1, eq.(55)
            q_of_qr: torch.Tensor = torch.linalg.qr(LEFT[:, -N:], mode="complete")[0]
            LEFT = q_of_qr[..., N:].conj().mT @ LEFT[..., :2 * M]
            RIGHT = q_of_qr[..., :2 * M, N:].conj().mT @ RIGHT[..., :2 * M, :2 * M]

            # U = torch.tensor(scipy.linalg.ordqz(
            #     LEFT.numpy(force=True), RIGHT.numpy(force=True),
            #     sort="lhp", output="real", overwrite_a=True, overwrite_b=True,
            # )[-1])
            def lhp(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
                return (np.real(alpha / beta) <= 0)

            AA, BB, alpha, beta, Q, U = map(torch.tensor, scipy.linalg.ordqz(
                LEFT.numpy(force=True), RIGHT.numpy(force=True),
                sort=lhp, output="real", overwrite_a=True, overwrite_b=True,
            ))
            
        U11 = U[..., :M, :M]
        U21 = U[..., M:2 * M, :M]
        P: torch.Tensor = U21 @ torch.linalg.pinv(U11)
    
    return P.to(original_dtype)

def test_continuous_are(
        A: torch.Tensor,
        B: torch.Tensor,
        Q: torch.Tensor,
        R: torch.Tensor,
        P: torch.Tensor,
) -> torch.Tensor:
    P = (P + P.mT) / 2
    ATP = A.mT @ P
    PB = P @ B
    return ATP + ATP.mT - PB @ torch.linalg.pinv(R) @ PB.mT + Q


"""
NumPy Array Comprehension Operations
"""

def multi_iter(arr: np.ndarray | DimArray) -> Iterable[Any]:
    for x in np.nditer(arr, flags=["refs_ok",],):
        yield x[()]

def multi_enumerate(arr: np.ndarray | DimArray) -> Iterable[Tuple[Sequence[int], Any]]:
    it = np.nditer(arr, flags=["multi_index", "refs_ok",],)
    for x in it:
        yield it.multi_index, x[()]

def multi_map(func: Callable[[Any], Any], arr: np.ndarray | DimArray, dtype: type = None):
    if dtype is None:
        dtype = type(func(arr.ravel()[0]))
    result = np.empty_like(arr, dtype=dtype)
    for idx, x in multi_enumerate(arr):
        result[idx] = func(x)
    return DimArray(result, dims=arr.dims) if isinstance(arr, DimArray) else result

def multi_zip(*arrs: np.ndarray) -> np.ndarray:
    result = np.recarray(arrs[0].shape, dtype=[(f"f{i}", arr.dtype) for i, arr in enumerate(arrs)])
    for i, arr in enumerate(arrs):
        setattr(result, f"f{i}", arr)
    return result


"""
DimArray Operations
"""

def dim_array_like(arr: DimArray, dtype: type) -> DimArray:
    empty_arr = np.full_like(arr, None, dtype=dtype)
    return DimArray(empty_arr, dims=arr.dims)

def broadcast_dim_array_shapes(*dim_arrs: Iterable[DimArray]) -> OrderedDict[str, int]:
    dim_dict = OrderedDict()
    for dim_arr in dim_arrs:
        for dim_name, dim_len in zip(dim_arr.dims, dim_arr.shape):
            dim_dict.setdefault(dim_name, []).append(dim_len)
    return OrderedDict((k, np.broadcast_shapes(*v)[0]) for k, v in dim_dict.items())

def broadcast_dim_arrays(*dim_arrs: Iterable[np.ndarray]) -> Iterator[DimArray]:
    _dim_arrs = []
    for dim_arr in dim_arrs:
        if isinstance(dim_arr, DimArray):
            _dim_arrs.append(dim_arr)
        elif isinstance(dim_arr, np.ndarray):
            assert dim_arr.ndim == 0
            _dim_arrs.append(DimArray(dim_arr, dims=[]))
        else:
            _dim_arrs.append(DimArray(array_of(dim_arr), dims=[]))
    dim_arrs = _dim_arrs

    dim_dict = broadcast_dim_array_shapes(*dim_arrs)
    reference_dim_arr = DimArray(
        np.zeros((*dim_dict.values(),)),
        dims=(*dim_dict.keys(),),
        axes=(*map(np.arange, dim_dict.values()),)
    )
    return (dim_arr.broadcast(reference_dim_arr) for dim_arr in dim_arrs)

def take_from_dim_array(dim_arr: DimArray | Dataset, idx: Dict[str, Any]):
    dims = set(dim_arr.dims)
    return dim_arr.take(indices={k: v for k, v in idx.items() if k in dims})


"""
Recursive attribute functions
"""
def rgetattr(obj: object, attr: str, *args):
    def _getattr(obj: object, attr: str) -> Any:
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split("."))

def rsetattr(obj: object, attr: str, value: Any) -> None:
    def _rsetattr(obj: object, attrs: List[str], value: Any) -> None:
        if len(attrs) == 1:
            setattr(obj, attrs[0], value)
        else:
            _rsetattr(next_obj := getattr(obj, attrs[0], Namespace()), attrs[1:], value)
            setattr(obj, attrs[0], next_obj)
    _rsetattr(obj, attr.split("."), value)

def rhasattr(obj: object, attr: str) -> bool:
    try:
        rgetattr(obj, attr)
        return True
    except AttributeError:
        return False

def rgetitem(obj: dict[str, Any], item: str, *args):
    def _getitem(obj: dict[str, Any], item: str) -> Any:
        return obj.get(item, *args)
    return functools.reduce(_getitem, [obj] + item.split("."))

def rsetitem(obj: dict[str, Any], item: str, value: Any) -> None:
    def _rsetitem(obj: dict[str, Any], items: List[str], value: Any) -> None:
        if len(items) == 1:
            obj[items[0]] = value
        else:
            _rsetitem(next_obj := obj.get(items[0], {}), items[1:], value)
            obj[items[0]] = next_obj
    _rsetitem(obj, item.split("."), value)


"""
Argument namespace processing
"""
class DefaultingParameter(Namespace):
    def __init__(self, default_key: str = TRAINING_DATASET_TYPES[0], **kwargs):
        Namespace.__init__(self, **kwargs)
        self._default_key = default_key

    def __getattr__(self, item):
        return vars(self).get(item, vars(self)[self._default_key])
    
    def default(self):
        return vars(self)[self._default_key]

    def update(self, **kwargs) -> None:
        vars(self).update(kwargs)

    def reset(self, **kwargs) -> None:
        vars(self).clear()
        vars(self).update(kwargs)

def process_defaulting_roots(o: _T) -> _T:
    ds_types = (*TRAINING_DATASET_TYPES, TESTING_DATASET_TYPE)
    if isinstance(o, Namespace):
        if len(vars(o)) > 0 and all(k in ds_types for k in vars(o)):
            return DefaultingParameter(**vars(o))
        else:
            for k, v in vars(o).items():
                setattr(o, k, process_defaulting_roots(v))
            return o
    else:
        return DefaultingParameter(**{TRAINING_DATASET_TYPES[0]: o})

def index_defaulting_with_attr(o: object, attr: str = None) -> Any:
    if isinstance(o, DefaultingParameter):
        return getattr(o, o._default_key if attr is None else attr)
    elif isinstance(o, Namespace):
        return Namespace(**{k: index_defaulting_with_attr(v, attr) for k, v in vars(o).items()})
    else:
        return o

def deepcopy_namespace(n: Namespace) -> Namespace:
    def _deepcopy_helper(o: _T) -> _T:
        if isinstance(o, Namespace):
            return type(o)(**{k: _deepcopy_helper(v) for k, v in vars(o).items()})
        else:
            return o
    return _deepcopy_helper(n)

def toJSON(o: object):
    if isinstance(o, Namespace):
        return {k: toJSON(v) for k, v in vars(o).items()}
    elif isinstance(o, dict):
        return {k: toJSON(v) for k, v in o.items()}
    elif isinstance(o, (list, tuple, set)):
        return list(map(toJSON, o))
    else:
        try:
            json.dumps(o)
            return o
        except TypeError:
            return str(o)

def str_namespace(n: Namespace) -> str:
    return json.dumps(toJSON(n), indent=4)

def print_namespace(n: Namespace) -> None:
    print(str_namespace(n))

def hash_namespace(n: Namespace) -> str:
    return hashlib.sha256(str_namespace(n).encode("utf-8")).hexdigest()[:8]

def hash_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]

def print_dict(d: dict[str, Any] | object, n: int = 0, indent: int = 4,) -> None:
    if isinstance(d, dict):
        for k, v in d.items():
            print(" " * (n * indent) + k)
            print_dict(v, n=n + 1, indent=indent,)
    else:
        to_print = str(d)
        print("\n".join([" " * (n * indent) + s for s in to_print.split("\n")]))

def td_get(d: TensorDict, keys: Sequence) -> TensorDict:
    return TensorDict({k: d[k] for k in keys}, batch_size=d.shape)


"""
Miscellaneous
"""
class Timer:
    def __init__(self):
        self.t = time.perf_counter()

    def reset(self):
        t = time.perf_counter()
        out = t - self.t
        self.t = t
        return out

def identity(x: Any) -> Any:
    return x
    
def reset_seed():
    if SEED is None:
        torch.seed()
        np.random.seed()
    else:
        torch.manual_seed(SEED)
        np.random.seed(SEED)

def torch_load(f: torch.types.FileLike, **kwargs) -> Any:
    return torch.load(f, map_location=DEVICE, weights_only=False,)

def empty_cache():
    gc.collect()
    torch.cuda.empty_cache()

class PTR(object):
    def __init__(self, obj: object) -> None:
        self.obj = obj

    def __iter__(self):
        yield self.obj

class default_dtype:
    def __init__(self, dtype: torch.dtype):
        self.dtype = dtype
    
    def __enter__(self):
        if self.dtype is not None:
            self._original_dtype = torch.get_default_dtype()
            torch.set_default_dtype(self.dtype)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.dtype is not None:
            torch.set_default_dtype(self._original_dtype)

def flatten_nested_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    def _flatten_nested_dict(s: Tuple[str, ...], d: Dict[str, Any]) -> None:
        for k, v in d.items():
            if isinstance(v, dict):
                _flatten_nested_dict((*s, k), v)
            else:
                result[".".join((*s, k))] = v
    _flatten_nested_dict((), d)
    return result

def nested_vars(n: Namespace) -> Dict[str, Any]:
    result = {}
    def _nested_vars(s: Tuple[str, ...], n: Namespace) -> None:
        for k, v in vars(n).items():
            if isinstance(v, Namespace):
                _nested_vars((*s, k), v)
            else:
                result[(*s, k)] = v
    _nested_vars((), n)
    return {".".join(k): v for k, v in result.items()}

def nested_type(o: object) -> object:
    if type(o) in [list, tuple]:
        return type(o)(map(nested_type, o))
    elif type(o) == dict:
        return {k: nested_type(v) for k, v in o.items()}
    else:
        return type(o)

def map_dict(d: Dict[str, Any], func: Callable[[Any], Any]) -> Dict[str, Any]:
    return {
        k: map_dict(v, func) if hasattr(v, "items") else func(v)
        for k, v in d.items()
    }

def array_of(o: _T) -> np.ndarray[_T]:
    M = np.array(None, dtype=object)
    M[()] = o
    return M

def model_size(m: nn.Module):
    return sum(p.numel() for p in m.parameters())

def call_func_with_kwargs(func: Callable, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
    params = inspect.signature(func).parameters
    required_args = [
        kwargs[k] if k in kwargs else args[i] for i, (k, v) in enumerate(params.items())
        if v.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD and v.default is inspect.Parameter.empty
    ]
    additional_args = args[len(required_args):]

    allow_var_keywords = any(v.kind is inspect.Parameter.VAR_KEYWORD for v in params.values())
    valid_kwargs = {
        k: v for k, v in kwargs.items()
        if ((params[k].default is not inspect.Parameter.empty) if k in params else allow_var_keywords)
    }
    return func(*required_args, *additional_args, **valid_kwargs)

def broadcast_shapes(*shapes: Tuple[int, ...]):
    def to_tuple(shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return (*map(int, shape),)
    return to_tuple(torch.broadcast_shapes(*map(to_tuple, shapes)))


""" Logging """
def get_tensors_in_memory(allowed_classes: Tuple[type] = (torch.Tensor,)) -> OrderedDict[int, torch.Tensor]:
    gc.collect()
    tensors = [obj for obj in gc.get_objects() if (type(obj) in allowed_classes) and (torch.is_tensor(obj) or torch.is_tensor(getattr(obj, "data", None)))]
    indices = np.argsort([t.numel() for t in tensors])[::-1]
    result = collections.OrderedDict()
    for idx in indices:
        t = tensors[idx]
        result[t.data_ptr()] = t
    return result

def print_tensors_in_memory(allowed_classes: Tuple[type] = (torch.Tensor,)) -> None:
    t_dict = get_tensors_in_memory(allowed_classes=allowed_classes)
    for t in t_dict.values():
        if t.numel() == 501:
            raise Exception("501")
        print(type(t), t.size(), t.numel())

def get_tensors_in_memory_shape(allowed_classes: Tuple[type] = (torch.Tensor,)) -> OrderedDict[int, torch.Size]:
    tensors = get_tensors_in_memory(allowed_classes)
    out = {k: v.shape for k, v in tensors.items()}
    del tensors
    gc.collect()
    return out

def track_tensor_diff(allowed_classes: Tuple[type] = (torch.Tensor,)) -> Iterator[tuple[list[tuple[int, torch.Size]], list[tuple[int, torch.Size]]]]:
    prev_tensors = get_tensors_in_memory_shape(allowed_classes)
    while True:
        tensors = get_tensors_in_memory_shape(allowed_classes)

        p = [(k, v) for k, v in tensors.items() if k not in prev_tensors]
        n = [(k, v) for k, v in prev_tensors.items() if k not in tensors]
        prev_tensors = tensors

        yield (p, n)

class print_disabled:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

class print_enabled:
    def __init__(self, enabled: bool):
        self.enabled = enabled
    
    def __enter__(self):
        if not self.enabled:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            sys.stdout.close()
            sys.stdout = self._original_stdout

def track_calls(desc=None, unit="call", log_args=None):
    def decorator(func):
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        pbar = tqdm(desc=desc or func.__name__, unit=unit)

        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            pbar.update(1)

            if log_args:
                postfix = {}
                for arg in log_args:
                    if isinstance(arg, int):
                        postfix[f"arg[{arg}]"] = args[arg] if arg < len(args) else "?"
                    elif arg in kwargs:
                        postfix[arg] = kwargs[arg]
                    elif arg in params:
                        idx = params.index(arg)
                        postfix[arg] = args[idx] if idx < len(args) else "?"
                pbar.set_postfix(postfix)

            return result

        wrapper.close = pbar.close
        return wrapper
    return decorator

def get_all_hooks(module: nn.Module) -> Dict[str, Dict[int, Callable]]:
    """
    Retrieves all forward and backward hooks, including pre-hooks, from a module and its submodules.

    Args:
        module: The nn.Module to inspect.

    Returns:
        A dictionary where keys are module names (or "" for the input module itself),
        and values are dictionaries of hook IDs to hook functions.
    """
    all_hooks = {}
    def _get_hooks(m: nn.Module, prefix=""):
        hooks = {}
        if hasattr(m, "_forward_hooks") and m._forward_hooks != OrderedDict():
            hooks.update({"forward_hooks": m._forward_hooks})
        if hasattr(m, "_forward_pre_hooks") and m._forward_pre_hooks != OrderedDict():
            hooks.update({"forward_pre_hooks": m._forward_pre_hooks})
        if hasattr(m, "_backward_hooks") and m._backward_hooks != OrderedDict():
             hooks.update({"backward_hooks": m._backward_hooks})
        if hasattr(m, "_full_backward_hooks") and m._full_backward_hooks != OrderedDict():
            hooks.update({"full_backward_hooks": m._full_backward_hooks})
        if hooks:
            all_hooks[prefix] = hooks

        for name, child in m.named_children():
            _get_hooks(child, prefix=f"{prefix}.{name}" if prefix else name)

    _get_hooks(module)
    return all_hooks


def _decode_png(png_bytes: bytes) -> np.ndarray:
    """Decode a PNG byte string to a uint8 RGB array of shape (H, W, 3).

    Uses imageio.v3 (which uses Pillow under the hood by default for PNG) and
    drops the alpha channel if present.  Always returns a contiguous array so
    downstream slicing in `_normalize` is cheap.
    """
    import io as _io
    import imageio.v3 as iio_v3
    arr = np.asarray(iio_v3.imread(_io.BytesIO(png_bytes), extension=".png"))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return np.ascontiguousarray(arr[..., :3])


def render_frames_to_mp4(
        recorded_frames: list[dict],
        key: str,
        out_path: Path,
        fps: float,
        playback_speed: float = 1.0,
        codec: str = "libx264",
) -> int:
    """Assemble recorded frames into an mp4, honoring inhomogeneous timestamps.

    Each output frame at video-time `v_j = j / fps` is mapped to the most recent
    recorded frame whose simulation time `t_i` satisfies `t_i <= t_0 + v_j *
    playback_speed`.  Recorded frames thus "stick" until a newer one arrives —
    a stretch of dense events shows quick changes, a sparse stretch freezes.

    `playback_speed` is the ratio (sim_dt / video_dt); `1.0` plays at real time,
    `0.5` plays at 2x slow-motion, `2.0` plays at 2x fast-forward.

    Returns the number of frames written.  Returns 0 (and writes nothing) if
    there are no valid frames for `key`.
    """
    valid = [(f["t"], f[key]) for f in recorded_frames if f.get(key) is not None]
    if not valid:
        return 0

    times = np.asarray([t for t, _ in valid], dtype=np.float64)
    pngs = [png for _, png in valid]
    t_start, t_end = float(times[0]), float(times[-1])

    sim_duration = max(t_end - t_start, 0.0)
    video_duration = sim_duration / max(playback_speed, 1e-12)
    n_frames = max(1, int(math.ceil(video_duration * fps)))

    # Bounded single-frame cache: `idx` is monotone non-decreasing in
    # the loop below (`searchsorted` on a sorted `times` with
    # monotone `sim_t`), so we never need to look back at a previously
    # decoded frame.  Keeping only the most-recent decode caps RAM at
    # ~one decoded RGBA buffer (~2 MB at 12×6"@100DPI).  Compare with
    # the legacy unbounded `dict[int, np.ndarray]` cache, which grew
    # to ~all consumed frames × ~2 MB ≈ 1-2 GB on a 10k-step run
    # (and ~3-4 GB when called twice for viz + hist back-to-back).
    cache_idx: int = -1
    cache_arr: np.ndarray | None = None
    def _frame(idx: int) -> np.ndarray:
        nonlocal cache_idx, cache_arr
        if idx == cache_idx and cache_arr is not None:
            return cache_arr
        cache_arr = _decode_png(pngs[idx])
        cache_idx = idx
        return cache_arr

    ref = _frame(0)
    H_ref = ref.shape[0] + (ref.shape[0] & 1)
    W_ref = ref.shape[1] + (ref.shape[1] & 1)

    def _normalize(arr: np.ndarray) -> np.ndarray:
        h, w = arr.shape[:2]
        if (h, w) == (H_ref, W_ref):
            return arr
        out = np.zeros((H_ref, W_ref, 3), dtype=arr.dtype)
        h_c, w_c = min(h, H_ref), min(w, W_ref)
        out[:h_c, :w_c] = arr[:h_c, :w_c]
        return out

    out_path.parent.mkdir(parents=True, exist_ok=True)
    import imageio.v3 as iio_v3

    with iio_v3.imopen(str(out_path), "w", plugin="pyav") as writer:
        writer.init_video_stream(codec, fps=fps)
        for j in range(n_frames):
            sim_t = t_start + (j / fps) * playback_speed
            idx = int(np.searchsorted(times, sim_t, side="right") - 1)
            idx = max(0, min(idx, len(valid) - 1))
            writer.write_frame(_normalize(_frame(idx)))
    return n_frames


""" Plotting code """
def color(z: float, scale: float = 120.) -> np.ndarray:
    k = 2 * np.pi * z / scale
    return (1 + np.asarray([np.sin(k), np.sin(k + 2 * np.pi / 3), np.sin(k + 4 * np.pi / 3)], dtype=float)) / 2

def confidence_ellipse(x, y, ax, n_std=1.0, facecolor="none", **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The Axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse"s radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    x, y = np.array(x), np.array(y)
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    M = np.stack([x, y], axis=0)
    cov = (M @ M.T) / len(x)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from the squareroot of the variance and multiplying with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std

    # Calculating the standard deviation of y
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def weighted_gaussian_kde_2d(
        points: torch.Tensor,
        weights: torch.Tensor,
        extent: Tuple[float, float, float, float],
        resolution: Tuple[int, int] = (256, 256,),
        min_bandwidth_px: float = 1.0,
        normalize: bool = True,
) -> torch.Tensor:
    """
    Weighted 2D Gaussian KDE evaluated on a regular grid.

    Drop-in replacement for the "draw as heatmap" use case of
    `scipy.stats.gaussian_kde`, but:
      1) runs on `points.device` (CUDA when points are on CUDA) via
         `scatter_add` + separable `F.conv2d`, so it is O(N + R_y * R_x * k)
         instead of scipy's O(N * R_y * R_x);
      2) uses Scott's rule on the weighted covariance for a diagonal
         bandwidth, but clamps each axis's σ below by `min_bandwidth_px`
         pixels. This prevents the degenerate case where a single dominant
         weight collapses the weighted covariance to ~0 (e.g. the particle
         filter at high velocities), which makes scipy's KDE bandwidth
         collapse too and renders a uniformly-zero grid.

    Parameters
    ----------
    points : float [N x 2]
        Point positions in `(y, x)` order (row-first, matching the default
        `imshow` orientation).
    weights : float [N]
        Non-negative weights. If `normalize=True` they are rescaled to sum
        to 1. NaNs and negatives are clamped to 0.
    extent : (y_min, y_max, x_min, x_max)
        Physical-axis extent of the grid. Suitable to pass straight into
        `imshow(..., extent=(x_min, x_max, y_max, y_min), origin="upper")`.
    resolution : (R_y, R_x)
        Grid size in cells along each axis.
    min_bandwidth_px : float
        Minimum Gaussian σ along each axis, in pixel units (i.e. bin widths).
    normalize : bool
        Controls the integral of the returned density: True -> integrates to
        ~1 (renormalize histogram weights to sum to 1), False -> integrates
        to ``Σ weights`` (raw mass per cell, the "PHD intensity" semantics).
        Bandwidth is *always* derived from a shape-only (scale-invariant)
        normalized copy of the weights, regardless of this flag — see below.

    Returns
    -------
    density : float [R_y x R_x]
        Density values on the grid. Integrates to ~1 when `normalize=True`;
        integrates to ``Σ weights`` when `normalize=False`.
    """
    P_DIM = 2
    device, dtype = points.device, points.dtype
    y_min, y_max, x_min, x_max = extent
    R_y, R_x = resolution
    dy, dx = (y_max - y_min) / R_y, (x_max - x_min) / R_x

    w = weights.to(dtype).nan_to_num(nan=0.0).clamp_min(0.0)

    if points.shape[0] == 0:
        return torch.zeros((R_y, R_x,), device=device, dtype=dtype)

    # Bandwidth math is *always* scale-invariant: weighted mean / variance /
    # effective sample size are properties of the *shape* of the weight
    # distribution, not of `Σ w`.  The original code applied them to `w`
    # directly, which was correct only when `Σ w ≈ 1`.  At larger total
    # mass (e.g., unnormalized PHD intensity at later filter steps) the
    # weighted "mean" lands far outside the image, `var_yx` blows up
    # quadratically, and Scott's rule then asks for a Gaussian kernel with
    # σ on the order of 10¹³ pixels — which downstream materializes as a
    # `torch.arange(-ry, ry+1)` of ~10¹⁰ elements (~138 GiB) and OOMs
    # immediately.  Normalizing here defends against that without changing
    # the bandwidth in the well-behaved `Σ w ≈ 1` case.
    w_for_bw = w / w.sum().clamp_min(torch.finfo(dtype).tiny)
    mean = (w_for_bw[..., None] * points).sum(dim=0)
    diff = points - mean
    var_yx = (w_for_bw[..., None] * diff * diff).sum(dim=0)                 # float: [2]
    neff = 1.0 / (w_for_bw * w_for_bw).sum().clamp_min(1e-12)
    scott2 = neff.pow(-2.0 / (P_DIM + 4))
    sigma_y = (var_yx[0].clamp_min(0) * scott2).sqrt().clamp_min(min_bandwidth_px * dy)
    sigma_x = (var_yx[1].clamp_min(0) * scott2).sqrt().clamp_min(min_bandwidth_px * dx)

    # `normalize` now only affects the histogram (and thus the integral of
    # the output density); reuse `w_for_bw` for free when the caller asked
    # for a unit-integral output.
    w_hist = w_for_bw if normalize else w
    iy = ((points[..., 0] - y_min) / dy).to(torch.long).clamp_(0, R_y - 1)
    ix = ((points[..., 1] - x_min) / dx).to(torch.long).clamp_(0, R_x - 1)
    hist = torch.zeros(R_y * R_x, device=device, dtype=dtype)
    hist.scatter_add_(0, iy * R_x + ix, w_hist)
    hist = hist.view(1, 1, R_y, R_x)

    sy_px = float((sigma_y / dy).item())
    sx_px = float((sigma_x / dx).item())
    # Belt-and-braces cap on kernel half-width: even after the bandwidth
    # fix above, an exotic weight distribution (e.g., a single dominant
    # particle near a corner) can still produce σ > image diagonal.  A
    # Gaussian wider than the grid behaves like a uniform smear regardless
    # of its true σ, so capping the conv kernel at the grid resolution is
    # mathematically lossless and keeps memory bounded at O(R_y · R_x).
    ry = max(1, min(int(math.ceil(3.0 * sy_px)), R_y))
    rx = max(1, min(int(math.ceil(3.0 * sx_px)), R_x))
    ys_1d = torch.arange(-ry, ry + 1, device=device, dtype=dtype)
    xs_1d = torch.arange(-rx, rx + 1, device=device, dtype=dtype)
    ker_y = torch.exp(-0.5 * (ys_1d / sy_px) ** 2); ker_y = ker_y / ker_y.sum()
    ker_x = torch.exp(-0.5 * (xs_1d / sx_px) ** 2); ker_x = ker_x / ker_x.sum()

    density = Fn.conv2d(hist, ker_y.view(1, 1, -1, 1), padding=(ry, 0))
    density = Fn.conv2d(density, ker_x.view(1, 1, 1, -1), padding=(0, rx))
    return density[0, 0] / (dx * dy)


def local_contrast_normalize_2d(
        density: torch.Tensor,
        sigma_px: float,
        eps_rel: float = 1e-3,
) -> torch.Tensor:
    """Local-contrast (DoG-style ratio) normalization of a 2D density grid.

    Returns ``density / max(gaussian_blur(density, σ_px), eps_rel · max(blur))``.
    A peak that is N× higher than its local neighborhood-average density
    saturates near N regardless of the global density scale, so secondary
    clusters in a multimodal posterior render at the same brightness as the
    global mode (vs straight `density / max(density)` where they fade in
    proportion to the global peak).  The denominator floor `eps_rel ·
    max(blur)` caps amplification at ~1/eps_rel in regions far from any
    particle (where the blurred density approaches 0) and prevents division-
    by-zero.

    Parameters
    ----------
    density : float [R_y, R_x]
        Non-negative density grid, typically the output of
        `weighted_gaussian_kde_2d`.
    sigma_px : float
        Standard deviation of the local-average Gaussian kernel, in
        grid-cell pixels.  Larger σ → larger neighborhood → coarser local
        contrast (asymptotes to global max-normalization as σ approaches
        the grid diagonal).  Smaller σ → tighter neighborhood, less change
        from the input.
    eps_rel : float
        Relative floor on the denominator, scaled by `max(blur)`.

    Returns
    -------
    ratio : float [R_y, R_x]
        Local-contrast normalized density on the same grid.
    """
    if density.numel() == 0:
        return density.clone()
    sigma_px = max(float(sigma_px), 1e-6)
    R_y, R_x = density.shape
    device, dtype = density.device, density.dtype

    # Cap the kernel half-width at the grid resolution; a Gaussian wider
    # than the grid behaves like a uniform smear regardless of true σ, and
    # this keeps memory bounded at O(R_y · R_x) for large σ.
    r = max(1, min(int(math.ceil(3.0 * sigma_px)), max(R_y, R_x)))
    coords = torch.arange(-r, r + 1, device=device, dtype=dtype)
    ker = torch.exp(-0.5 * (coords / sigma_px) ** 2)
    ker = ker / ker.sum()

    inp = density.view(1, 1, R_y, R_x)
    blurred = Fn.conv2d(inp, ker.view(1, 1, -1, 1), padding=(r, 0))
    blurred = Fn.conv2d(blurred, ker.view(1, 1, 1, -1), padding=(0, r))
    blurred = blurred[0, 0]

    blur_max = float(blurred.max().item())
    floor = blur_max * float(eps_rel)
    if floor <= 0.0:
        floor = float(torch.finfo(dtype).tiny)
    return density / blurred.clamp_min(floor)


def vector_weighted_gaussian_kde_2d(
        points: torch.Tensor,
        scalar_w: torch.Tensor,
        vector_w: torch.Tensor,
        extent: Tuple[float, float, float, float],
        resolution: Tuple[int, int] = (256, 256,),
        sigma_px: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Joint scalar + vector Gaussian smoothing on a regular 2D grid.

    Computes
        M(g) = Σ_i scalar_w_i  · K_σ(g − p_i)
        V(g) = Σ_i vector_w_i · K_σ(g − p_i)
    on the same grid via a single shared scatter and a single separable
    Gaussian convolution (in `1 + C` channels).  The Nadaraya–Watson estimate
    of any quantity stored in `vector_w / scalar_w` (e.g., the smoothed
    velocity field given `scalar_w = w_i`, `vector_w = w_i v_i`) is `V / M`.

    Unlike `weighted_gaussian_kde_2d`, this helper takes a *fixed* user-set
    bandwidth (no Scott's rule, no `min_bandwidth_px` floor).  This is the
    desired knob for visualization smoothing whose size should be decoupled
    from any data-driven detection / display bandwidth.

    Output is integrated mass per cell (no `1 / (dx · dy)` density factor)
    because the only intended downstream operation is the ratio `V / M`,
    where the cell-area factor would cancel anyway.  Thresholding `M` is
    therefore done in raw mass units, which is more interpretable.

    Parameters
    ----------
    points : float [N x 2]
        `(y, x)` particle positions.
    scalar_w : float [N]
        Per-particle scalar weights (e.g., `softmax(log_w)`).  NaNs and
        negatives are clamped to 0.
    vector_w : float [N x C]
        Per-particle vector weights (e.g., `scalar_w[..., None] *
        velocity_i`, in `(y, x)` order to match `points`).  NaNs are
        replaced with 0; signs are preserved.
    extent : (y_min, y_max, x_min, x_max)
        Physical extent of the grid.
    resolution : (R_y, R_x)
        Grid size in cells along each axis.
    sigma_px : float
        Gaussian σ in *grid-cell* pixels (same on both axes).

    Returns
    -------
    M : float [R_y, R_x]
        Smoothed scalar mass per cell.
    V : float [R_y, R_x, C]
        Smoothed vector moment per cell.
    """
    device, dtype = points.device, points.dtype
    y_min, y_max, x_min, x_max = extent
    R_y, R_x = resolution
    dy, dx = (y_max - y_min) / R_y, (x_max - x_min) / R_x
    C = vector_w.shape[-1]

    if points.shape[0] == 0:
        return (
            torch.zeros((R_y, R_x,), device=device, dtype=dtype),
            torch.zeros((R_y, R_x, C,), device=device, dtype=dtype),
        )

    sw = scalar_w.to(dtype).nan_to_num(nan=0.0).clamp_min(0.0)
    vw = vector_w.to(dtype).nan_to_num(nan=0.0)

    iy = ((points[..., 0] - y_min) / dy).to(torch.long).clamp_(0, R_y - 1)
    ix = ((points[..., 1] - x_min) / dx).to(torch.long).clamp_(0, R_x - 1)
    flat_idx = iy * R_x + ix                                            # [N]

    # One scatter per channel, batched as the leading dim.  Each "image"
    # is single-channel, so the subsequent conv runs with groups=1 and the
    # batched images go through together.
    channels = torch.cat((sw[..., None], vw,), dim=-1).t().contiguous() # [1+C, N]
    hist = torch.zeros((1 + C, R_y * R_x,), device=device, dtype=dtype)
    hist.scatter_add_(1, flat_idx[None, :].expand(1 + C, -1), channels)
    hist = hist.view(1 + C, 1, R_y, R_x)

    r = max(1, int(math.ceil(3.0 * sigma_px)))
    s_1d = torch.arange(-r, r + 1, device=device, dtype=dtype)
    ker = torch.exp(-0.5 * (s_1d / sigma_px) ** 2)
    ker = ker / ker.sum()
    smoothed = Fn.conv2d(hist, ker.view(1, 1, -1, 1), padding=(r, 0))
    smoothed = Fn.conv2d(smoothed, ker.view(1, 1, 1, -1), padding=(0, r))
    smoothed = smoothed[:, 0]                                           # [1+C, R_y, R_x]

    M = smoothed[0]
    V = smoothed[1:].permute(1, 2, 0).contiguous()                      # [R_y, R_x, C]
    return M, V


def mass_weighted_farthest_point_sampling(
        positions: torch.Tensor,
        mass: torch.Tensor,
        n: int,
) -> torch.Tensor:
    """Greedy farthest-point sampling weighted by per-candidate mass.

    Iteratively picks indices to maximize `mass[i] · min_sq_dist_to_selected[i]`,
    seeding with `argmax(mass)`.  At the selection threshold, `m · d² ≈ const`
    so `d ∝ 1/√m` and the 2D point density of the result is asymptotically
    proportional to `mass`.  Deterministic given the inputs (no randomness),
    which keeps frame-to-frame placement stable in video.

    Parameters
    ----------
    positions : float [N x D]
        Candidate positions (any D ≥ 1; only pairwise squared distances are
        used).
    mass : float [N]
        Non-negative per-candidate mass.  Candidates with mass = 0 will not
        be picked unless `n ≥ N` (in which case they appear at the tail with
        score 0 and the loop returns early, so the actual returned count
        reflects only positive-score picks).
    n : int
        Maximum number of points to return.

    Returns
    -------
    indices : long [k]
        Selected indices into the input tensors, in selection order.
        `k = min(n, N)` minus any tail picks whose score reached 0.

    Notes
    -----
    The loop body is sync-free: indices and per-pick scores accumulate into
    pre-allocated GPU buffers and the early-exit-on-score-≤-0 condition is
    deferred to a single host transfer at the end.  This keeps a 80-pick run
    at ~1 host sync instead of ~160 (`int(argmax)` + `float(score[idx])`
    every iteration).  The full `n` iterations always execute — picks past
    the original break point are still well-defined argmax choices, and the
    final prune restores bit-equivalent semantics for the kept prefix.
    """
    device = positions.device
    N = positions.shape[0]
    if N == 0 or n <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    n = min(n, N)

    selected = torch.empty(n, dtype=torch.long, device=device)
    pick_score = torch.empty(n, dtype=mass.dtype, device=device)

    first_idx = mass.argmax()
    selected[0] = first_idx
    # `pick_score[0]` is unused by the prune logic (the seed pick is always
    # kept, matching the original `selected = [first]` initialization), but
    # we still write a value so any caller that exposes the buffer sees a
    # well-defined number rather than uninitialized memory.
    pick_score[0] = mass[first_idx]
    delta = positions - positions[first_idx]
    min_sq_dist = (delta * delta).sum(-1)                               # float: [N]

    for k in range(1, n):
        score = mass * min_sq_dist
        idx = score.argmax()
        selected[k] = idx
        pick_score[k] = score[idx]
        delta = positions - positions[idx]
        new_sq_dist = (delta * delta).sum(-1)
        min_sq_dist = torch.minimum(min_sq_dist, new_sq_dist)

    if n == 1:
        return selected

    # Find the first k ≥ 1 where pick_score[k] ≤ 0; keep [0, k).  All
    # subsequent picks are dropped — once the FPS objective hits zero the
    # remaining candidates are duplicates / zero-mass and the original
    # `break` short-circuits the same prefix.  `argmax` over a long-dtype
    # mask returns the FIRST 1, exactly the index we want.
    nonpositive_tail = (pick_score[1:] <= 0).to(torch.long)
    has_invalid = nonpositive_tail.sum() > 0
    first_invalid = nonpositive_tail.argmax()                           # long, scalar
    k_keep = torch.where(
        has_invalid,
        first_invalid + 1,
        torch.tensor(n, dtype=torch.long, device=device),
    )
    # Single host sync for the slice length; everything else stayed on device.
    return selected[: int(k_keep.item())]


def bilinear_sample_2d(
        grid: torch.Tensor,
        points: torch.Tensor,
        extent: Tuple[float, float, float, float],
) -> torch.Tensor:
    """Sample a 2D scalar / vector field at continuous query points.

    Thin wrapper around `torch.nn.functional.grid_sample` with
    `align_corners=False` and `padding_mode='border'`.  This treats grid
    cells as squares (so cell `(i, j)` covers physical
    `[y_min + i·dy, y_min + (i+1)·dy) × …`) and clamps queries outside the
    extent to the boundary value (avoids spurious zeros where particles
    drift slightly past the frame edge during evolution).

    Parameters
    ----------
    grid : float [R_y, R_x]  or  [R_y, R_x, C]
        Field values on a regular grid spanning `extent`.
    points : float [N, 2]
        Query positions in `(y, x)` physical coords.
    extent : (y_min, y_max, x_min, x_max)
        Physical extent the grid covers.

    Returns
    -------
    values : float [N]  or  [N, C]
        Same trailing shape as `grid` minus the spatial dims.
    """
    y_min, y_max, x_min, x_max = extent
    ny = (points[..., 0] - y_min) / (y_max - y_min) * 2 - 1
    nx = (points[..., 1] - x_min) / (x_max - x_min) * 2 - 1
    # grid_sample expects (x, y) order in the last dim; shape [B, H_out, W_out, 2].
    sampling_grid = torch.stack((nx, ny,), dim=-1)[None, None, :, :]    # [1, 1, N, 2]

    if grid.ndim == 2:
        grid_4d = grid[None, None]                                      # [1, 1, R_y, R_x]
        squeeze_channel = True
    elif grid.ndim == 3:
        grid_4d = grid.permute(2, 0, 1)[None]                           # [1, C, R_y, R_x]
        squeeze_channel = False
    else:
        raise ValueError(f"`grid.ndim` must be 2 or 3, got {grid.ndim}")

    sampled = Fn.grid_sample(
        grid_4d, sampling_grid,
        mode="bilinear", padding_mode="border", align_corners=False,
    )                                                                   # [1, C, 1, N]
    sampled = sampled[0, :, 0]                                          # [C, N]
    if squeeze_channel:
        return sampled[0]                                               # [N]
    return sampled.t().contiguous()                                     # [N, C]




