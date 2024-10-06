import jax
import jax.numpy as jnp
import qutip as qu
import numpy as np

import equinox as eqx
from jax.scipy.linalg import expm


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


pauli_projective_povm = (
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


pauli_projective_povm_super = jnp.array(
    [a.flatten() for a in pauli_projective_povm.reshape(6, 2, 2)]
).reshape(-1, 2, 4)


def _make_pauli_dissipator(A, B):
    return _sprepost(A, B) - 0.5 * (_spre(B @ A) + _spost(B @ A))


dissipators_list = []
for g_i in _G[1:]:
    aux_list = []
    for g_j in _G[1:]:
        aux_list.append(_make_pauli_dissipator(g_i, g_j))
    dissipators_list.append(aux_list)

_Pauli_dissipators_array = jnp.array(dissipators_list, jnp.complex64)

generators_traceless_hermitian = jnp.array(
    [qu.sigmax().full(), qu.sigmay().full(), qu.sigmaz().full()]
)

generators_hermitian_3d_matrices = np.array(
    [
        np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
        np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
        np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]]),
        np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]]),
        np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]]),
        np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
        1j * np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]]),
        1j * np.array([[0, 0, -1], [0, 0, 0], [1, 0, 0]]),
        1j * np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]]),
    ]
)


@jax.jit
def generate_hamiltonian_one_qubit(
    hamiltonian_parameters, generators_traceless_hermitian
):
    return jnp.einsum("i, ijk", hamiltonian_parameters, generators_traceless_hermitian)


@jax.jit
def generate_hermitian_matrix(parameters, generators_hermitian):
    return jnp.einsum("i, ijk", parameters, generators_hermitian)


def _make_dissipator(dissipator_matrix, pauli_dissipators):
    return jnp.einsum("ij, ijmn-> mn", dissipator_matrix, pauli_dissipators)


def _make_superop_hamiltonian(hamiltonian_matrix, hbar=1):
    return -1j / hbar * (_spre(hamiltonian_matrix) - _spost(hamiltonian_matrix))


is_probability_correct = lambda p: jnp.logical_and((p >= 0.0), (p <= 1.0))
trim_invalid_probs = lambda prob_array: jnp.where(
    is_probability_correct(prob_array), prob_array, jnp.abs(prob_array) * 0
)

trim_nan_probs = lambda prob_array: jnp.where(
    ~jnp.isnan(prob_array), prob_array, jnp.abs(prob_array) * 0
)


def clean_probabilities(prob_array):
    return trim_nan_probs(trim_invalid_probs(prob_array))


def evolve_state(lindbladian, time, rho_super):
    return expm(lindbladian * time) @ rho_super


def compute_probability(rho_super, povm_super):
    return clean_probabilities(jnp.dot(_dag(rho_super), povm_super)).real


def generate_complete_lindbladian(parameters_hamiltonian, parameters_dissipator):
    hamiltonian = generate_hamiltonian_one_qubit(
        parameters_hamiltonian, generators_traceless_hermitian
    )
    lindblad_matrix = generate_hermitian_matrix(
        parameters_dissipator, generators_hermitian_3d_matrices
    )

    hamiltonian_superop = _make_superop_hamiltonian(hamiltonian)
    dissipator_superop = _make_dissipator(lindblad_matrix, _Pauli_dissipators_array)
    lindbladian = hamiltonian_superop + dissipator_superop
    return lindbladian


_set_of_initial_states_super = jnp.array(
    [
        qu.ket2dm(ket).full().flatten()
        for ket in [
            qu.basis(2, 0),
            qu.basis(2, 1),
            (qu.basis(2, 0) + qu.basis(2, 1)).unit(),
            (qu.basis(2, 0) + 1j * qu.basis(2, 1)).unit(),
        ]
    ]
)


def likelihood_experiment(experiment, parameters):
    initial_state_index = experiment.initial_state.squeeze()
    measurement_basis_index = experiment.measurement_basis.squeeze()
    time = experiment.time.squeeze()

    initial_state_super = _set_of_initial_states_super[initial_state_index]
    povm_super = pauli_projective_povm_super[measurement_basis_index]

    lindbladian = generate_complete_lindbladian(
        parameters.hamiltonian_pars, parameters.dissipator_pars
    )

    evolved_initial_state_super = evolve_state(lindbladian, time, initial_state_super)

    probabilities_outcomes_basis = compute_probability(
        evolved_initial_state_super, povm_super
    )
    return probabilities_outcomes_basis


def likelihood_data(data, parameters):
    experiment = data.experiment
    outcome = data.outcome.squeeze()
    probability_outcome = likelihood_experiment(experiment, parameters)[outcome]
    return probability_outcome


def neg_log_likelihood_data(data, parameters):
    minus_log_lkl = -1 * jnp.log(likelihood_data(data, parameters))
    return minus_log_lkl
