from typing import Callable, Optional

import numba
from numba import cuda

from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

# This code will CUDA compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.

to_index = cuda.jit(device=True)(to_index)
index_to_position = cuda.jit(device=True)(index_to_position)
broadcast_index = cuda.jit(device=True)(broadcast_index)

THREADS_PER_BLOCK = 32


class CudaOps(TensorOps):
    cuda = True

    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"
        f = tensor_map(cuda.jit(device=True)(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            # Instantiate and run the cuda kernel.
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = out.size + THREADS_PER_BLOCK - 1  # THREADS_PER_BLOCK
            f[blockspergrid, threadsperblock](*out.tuple(), out.size, *a.tuple())  # type: ignore
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        f = tensor_zip(cuda.jit(device=True)(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            threadsperblock = THREADS_PER_BLOCK
            blockspergrid = out.size + (threadsperblock - 1)  # threadsperblock
            f[blockspergrid, threadsperblock](  # type: ignore
                *out.tuple(), out.size, *a.tuple(), *b.tuple()
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        f = tensor_reduce(cuda.jit(device=True)(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = (a.shape[dim] - 1) // 1024 + 1
            out_a = a.zeros(tuple(out_shape))

            threadsperblock = 1024
            blockspergrid = out_a.size
            f[blockspergrid, threadsperblock](  # type: ignore
                *out_a.tuple(), out_a.size, *a.tuple(), dim, start
            )

            return out_a

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # One block per batch, extra rows, extra col
        blockspergrid = (
            (out.shape[1] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            (out.shape[2] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK,
            out.shape[0],
        )
        threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)

        tensor_matrix_multiply[blockspergrid, threadsperblock](
            *out.tuple(), out.size, *a.tuple(), *b.tuple()
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implement


def tensor_map(
    fn: Callable[[float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """
    CUDA higher-order tensor map function. ::

      fn_map = tensor_map(fn)
      fn_map(out, ... )

    Args:
        fn: function mappings floats-to-floats to apply.

    Returns:
        Tensor map function.
    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        in_idx = cuda.local.array(MAX_DIMS, numba.int32)
        out_idx = cuda.local.array(MAX_DIMS, numba.int32)

        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if i >= out_size:
            return

        to_index(i, out_shape, out_idx)
        broadcast_index(out_idx, out_shape, in_shape, in_idx)

        in_pos = index_to_position(in_idx, in_strides)
        out_pos = index_to_position(out_idx, out_strides)

        out[out_pos] = fn(in_storage[in_pos])

    return cuda.jit()(_map)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float]
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """
    CUDA higher-order tensor zipWith (or map2) function ::

      fn_zip = tensor_zip(fn)
      fn_zip(out, ...)

    Args:
        fn: function mappings two floats to float to apply.

    Returns:
        Tensor zip function.
    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:

        a_idx = cuda.local.array(MAX_DIMS, numba.int32)
        b_idx = cuda.local.array(MAX_DIMS, numba.int32)
        out_idx = cuda.local.array(MAX_DIMS, numba.int32)

        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if i >= out_size:
            return

        to_index(i, out_shape, out_idx)
        out_pos = index_to_position(out_idx, out_strides)

        broadcast_index(out_idx, out_shape, a_shape, a_idx)
        a_pos = index_to_position(a_idx, a_strides)

        broadcast_index(out_idx, out_shape, b_shape, b_idx)
        b_pos = index_to_position(b_idx, b_strides)

        out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])

    return cuda.jit()(_zip)  # type: ignore


def _sum_practice(out: Storage, a: Storage, size: int) -> None:
    """
    This is a practice sum kernel to prepare for reduce.

    Given an array of length $n$ and out of size $n // \text{blockDIM}$
    it should sum up each blockDim values into an out cell.

    $[a_1, a_2, ..., a_{100}]$

    |

    $[a_1 +...+ a_{31}, a_{32} + ... + a_{64}, ... ,]$

    Note: Each block must do the sum using shared memory!

    Args:
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        size (int):  length of a.

    """
    BLOCK_DIM = 32

    block_mem = cuda.shared.array(BLOCK_DIM, numba.float64)

    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i >= size:
        return

    block_mem[cuda.threadIdx.x] = a[i]

    cuda.syncthreads()

    if cuda.threadIdx.x == 0:
        tmp = cuda.local.array(shape=1, dtype=numba.float32)
        for i in range(BLOCK_DIM):
            tmp[0] += block_mem[i]
        out[cuda.blockIdx.x] = tmp[0]


jit_sum_practice = cuda.jit()(_sum_practice)


def sum_practice(a: Tensor) -> TensorData:
    (size,) = a.shape
    threadsperblock = THREADS_PER_BLOCK
    blockspergrid = (size // THREADS_PER_BLOCK) + 1
    out = TensorData([0.0 for i in range(2)], (2,))
    out.to_cuda_()
    jit_sum_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, size
    )
    return out


def tensor_reduce(
    fn: Callable[[float, float], float]
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """
    CUDA higher-order tensor reduce function.

    Args:
        fn: reduction function maps two floats to float.

    Returns:
        Tensor reduce function.

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        out_size: int,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
        reduce_value: float,
    ) -> None:
        BLOCK_DIM = 1024

        block_mem = cuda.shared.array(BLOCK_DIM, dtype=numba.float32)
        a_idx = cuda.local.array(MAX_DIMS, dtype=numba.int32)

        idx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if idx >= out_size:
            return

        to_index(idx, a_shape, a_idx)
        a_pos = index_to_position(a_idx, a_strides)

        block_mem[cuda.threadIdx.x] = a_storage[a_pos]

        cuda.syncthreads()

        if cuda.threadIdx.x == 0:
            tmp = cuda.local.array(shape=1, dtype=numba.float32)
            for i in range(BLOCK_DIM):
                tmp[0] += block_mem[i]

            out_idx = a_idx
            out_idx[reduce_dim] = 0
            out_pos = index_to_position(out_idx, out_strides)
            out[out_pos] = fn(out[out_pos], tmp[0])

    return cuda.jit()(_reduce)  # type: ignore


def _mm_practice(out: Storage, a: Storage, b: Storage, size: int) -> None:
    """
    This is a practice square MM kernel to prepare for matmul.

    Given a storage `out` and two storage `a` and `b`. Where we know
    both are shape [size, size] with strides [size, 1].

    Size is always < 32.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Compute

    ```
     for i:
         for j:
              for k:
                  out[i, j] += a[i, k] * b[k, j]
    ```

    Args:
        out (Storage): storage for `out` tensor.
        a (Storage): storage for `a` tensor.
        b (Storage): storage for `b` tensor.
        size (int): size of the square
    """
    shm_a = cuda.shared.array((THREADS_PER_BLOCK, THREADS_PER_BLOCK), numba.float64)
    shm_b = cuda.shared.array((THREADS_PER_BLOCK, THREADS_PER_BLOCK), numba.float64)

    idx_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    idx_y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if idx_x >= size or idx_y >= size:
        return

    pos = index_to_position((idx_x, idx_y), (size, 1))
    shm_a[idx_x][idx_y] = a[pos]
    shm_b[idx_x][idx_y] = b[pos]

    cuda.syncthreads()

    total = 0.0
    for i in range(size):
        total += shm_a[idx_x][i] * shm_b[i][idx_y]

    out[pos] = total


jit_mm_practice = cuda.jit()(_mm_practice)


def mm_practice(a: Tensor, b: Tensor) -> TensorData:
    (size, _) = a.shape
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = 1
    out = TensorData([0.0 for i in range(size * size)], (size, size))
    out.to_cuda_()
    jit_mm_practice[blockspergrid, threadsperblock](
        out.tuple()[0], a._tensor._storage, b._tensor._storage, size
    )
    return out


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    out_size: int,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """
    CUDA tensor matrix multiply function.

    Requirements:

    * All data must be first moved to shared memory.
    * Only read each cell in `a` and `b` once.
    * Only write to global memory once per kernel.

    Should work for any tensor shapes that broadcast as long as ::

    ```python
    assert a_shape[-1] == b_shape[-2]
    ```
    Returns:
        None : Fills in `out`
    """
    idx_x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    idx_y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    idx_z = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z

    shm_c = cuda.shared.array((THREADS_PER_BLOCK, THREADS_PER_BLOCK), numba.float64)
    shm_c[cuda.threadIdx.x][cuda.threadIdx.y] = 0.0

    shm_a = cuda.shared.array((THREADS_PER_BLOCK, THREADS_PER_BLOCK), numba.float64)
    shm_b = cuda.shared.array((THREADS_PER_BLOCK, THREADS_PER_BLOCK), numba.float64)
    count = (a_shape[-1] + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    for i in range(count):
        x_a = cuda.blockIdx.x * THREADS_PER_BLOCK + cuda.threadIdx.x
        y_a = i * THREADS_PER_BLOCK + cuda.threadIdx.y
        z_a = idx_z if out_shape[0] == a_shape[0] else 0
        if x_a < a_shape[1] and y_a < a_shape[2]:
            pos_a = index_to_position((z_a, x_a, y_a), a_strides)
            shm_a[cuda.threadIdx.x][cuda.threadIdx.y] = a_storage[pos_a]
        else:
            shm_a[cuda.threadIdx.x][cuda.threadIdx.y] = 0.0

        x_b = i * THREADS_PER_BLOCK + cuda.threadIdx.x
        y_b = cuda.blockIdx.y * THREADS_PER_BLOCK + cuda.threadIdx.y
        z_b = idx_z if out_shape[0] == b_shape[0] else 0
        if x_b < b_shape[1] and y_b < b_shape[2]:
            pos_b = index_to_position((z_b, x_b, y_b), b_strides)
            shm_b[cuda.threadIdx.x][cuda.threadIdx.y] = b_storage[pos_b]
        else:
            shm_b[cuda.threadIdx.x][cuda.threadIdx.y] = 0.0

        cuda.syncthreads()

        for j in range(THREADS_PER_BLOCK):
            shm_c[cuda.threadIdx.x][cuda.threadIdx.y] += (
                shm_a[cuda.threadIdx.x][j] * shm_b[j][cuda.threadIdx.y]
            )

        cuda.syncthreads()

    if idx_z < out_shape[0] and idx_x < out_shape[1] and idx_y < out_shape[2]:
        pos = index_to_position((idx_z, idx_x, idx_y), out_strides)
        out[pos] = shm_c[cuda.threadIdx.x][cuda.threadIdx.y]


tensor_matrix_multiply = cuda.jit(_tensor_matrix_multiply)
