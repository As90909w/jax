You can mix `jit` and `grad` and any other JAX transformation however you like.

Using `jit` puts constraints on the kind of Python control flow
the function can use; see
the [Gotchas
Notebook](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#python-control-flow-+-JIT)
for more.

### Auto-vectorization with `vmap`

[`vmap`](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap) is
the vectorizing map.
It has the familiar semantics of mapping a function along array axes, but
instead of keeping the loop on the outside, it pushes the loop down into a
function’s primitive operations for better performance.

Using `vmap` can save you from having to carry around batch dimensions in your
code. For example, consider this simple *unbatched* neural network prediction
function:

```python
def predict(params, input_vec):
  assert input_vec.ndim == 1
   primitives](https://jax.readthedocs.io/en/latest/jax.lax.html#control-flow-operators)
   like
   [`lax.scan`](https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html#jax.lax.scan),
   or just use `jit` on smaller subfunctions.
  ```
## Installation

### Supported platforms

|            | Linux x86_64 | Linux aarch64 | Mac x86_64   | Mac ARM      | Windows x86_64 | Windows WSL2 x86_64 |
|------------|--------------|---------------|--------------|--------------|----------------|---------------------|
| CPU        | yes          | yes           | yes          | yes          | yes            | yes                 |
| NVIDIA GPU | yes          | yes           | no           | n/a          | no             | experimental        |
| Google TPU | yes          | n/a           | n/a          | n/a          | n/a            | n/a                 |
| AMD GPU    | experimental | no            | no           | n/a          | no             | no                  |
| Apple GPU  | n/a          | no            | experimental | experimental | n/a            | n/a                 |


### Instructions

| Hardware   | Instructions                                                                                                    |
|------------|-----------------------------------------------------------------------------------------------------------------|
| CPU        | `pip install -U "jax[cpu]"`                                                                                       |
| NVIDIA GPU on x86_64 | `pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`        |
| Google TPU | `pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html`                 |
| AMD GPU    | Use [Docker](https://hub.docker.com/r/rocm/jax) or [build from source](https://jax.readthedocs.io/en/latest/developer.html#additional-notes-for-building-a-rocm-jaxlib-for-amd-gpus). |
| Apple GPU  | Follow [Apple's instructions](https://developer.apple.com/metal/jax/).                                          |

See [the documentation](https://jax.readthedocs.io/en/latest/installation.html)
for information on alternative installation strategies. These include compiling
from source, installing with Docker, using other versions of CUDA, a
community-supported conda build, and answers to some frequently-asked questions.





