import jax
import jax.numpy as jnp
import qutip as qu

from jax import jit


_G = jnp.array(
    [
        jnp.array(
            [[1, 0], [0, 1]],
        ),
        jnp.array(
            [[0, 1], [1, 0]],
        ),
        jnp.array(
            [[0, -1j], [1j, 0]],
        ),
        jnp.array(
            [[1, 0], [0, -1]],
        ),
    ]
) / jnp.sqrt(2)


@jit
def _dag(A):
    return jnp.conjugate(A.T)


@jit
def _sprepost(A, B):
    return jnp.kron(A, B.T)


@jit
def _spre(A):
    d = A.shape[0]
    return _sprepost(A, jnp.identity(d))


@jit
def _spost(A):
    d = A.shape[0]

    return _sprepost(jnp.identity(d), A)


@jit
def _vec(A):
    return A.flatten()


canonical_povm = (
    jnp.array(
        [
            qu.identity(2).full() + qu.sigmax().full(),
            qu.identity(2).full() - qu.sigmax().full(),
            qu.identity(2).full() + qu.sigmay().full(),
            qu.identity(2).full() - qu.sigmay().full(),
            qu.identity(2).full() + qu.sigmaz().full(),
            qu.identity(2).full() - qu.sigmaz().full(),
        ]
    )
    / 2
).reshape(-1, 2, 2, 2)
