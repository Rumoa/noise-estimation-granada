import jax
import jax.numpy as jnp
import equinox as eqx


class OneQubitParameters(eqx.Module):
    d: int
    N: int
    parameters: jnp.array

    def __init__(self, dimension_system, pars):
        self.d = dimension_system
        self.N = self.d**2
        self.parameters = self.set_pars(pars)

    @property
    def n_indep_hamiltonian(self):
        return self.d**2 - 1

    @property
    def n_indep_dissipator(self):
        return (self.N - 1) ** 2

    def set_pars(self, pars):
        parameters = jnp.zeros([self.n_indep_dissipator + self.n_indep_hamiltonian])
        parameters = parameters.at[:].set(pars)
        return parameters

    @property
    def hamiltonian_pars(self):
        return self.parameters[0 : self.n_indep_hamiltonian]

    @property
    def dissipator_pars(self):
        return self.parameters[self.n_indep_hamiltonian :]
