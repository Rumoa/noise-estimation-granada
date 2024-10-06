import equinox as eqx
import jax
from jax import numpy as jnp


def ensure_array(array):
    array = jnp.array(array)
    if array.shape == ():
        return jnp.array([array])
    return array


class Experiment(eqx.Module):
    pass


class ExperimentOneQubitTomography(Experiment):
    time: float
    initial_state: int
    measurement_basis: int

    def __init__(self, t: float, initial_state: int, measurement_basis: int) -> None:
        self.time = jnp.float32(ensure_array(t))
        self.initial_state = jnp.int8(ensure_array(initial_state))
        self.measurement_basis = jnp.int8(ensure_array(measurement_basis))

    def __len__(self):
        return len(self.time)

    def __iter__(self):
        for i in range(len(self)):
            yield ExperimentOneQubitTomography(
                self.time[i], self.initial_state[i], self.measurement_basis[i]
            )
        # for t in self.time:
        #     yield Experiment(t)

    def __getitem__(self, item):
        return ExperimentOneQubitTomography(
            self.time[item], self.initial_state[item], self.measurement_basis[item]
        )

    def __str__(self):
        s = f"Time {self.time}\nInitial state {self.initial_state}\nMeasurement basis {self.measurement_basis} "
        return s
