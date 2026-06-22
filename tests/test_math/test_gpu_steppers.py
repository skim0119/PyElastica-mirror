import jax
import jax.numpy as jnp
import numpy as np
import pytest

from elastica.timestepper.gpu_steppers import PositionVerletGPU


class FakeJAXBlock:
    def __init__(self):
        self.final_state = None

    def jax_get_state(self):
        return {
            "position": jnp.asarray(0.0),
            "velocity": jnp.asarray(1.0),
            "acceleration": jnp.asarray(2.0),
            "external_force": jnp.asarray(5.0),
            "internal_force": jnp.asarray(0.0),
        }

    def jax_set_state(self, state):
        self.final_state = state

    def jax_kinematic_step(self, state, time, prefac):
        updated = dict(state)
        updated["position"] = state["position"] + prefac * state["velocity"]
        return updated

    def jax_dynamic_step(self, state, time, dt):
        updated = dict(state)
        updated["velocity"] = state["velocity"] + dt * state["acceleration"]
        return updated

    def jax_compute_internal_forces_and_torques(self, state, time):
        updated = dict(state)
        updated["internal_force"] = time
        return updated

    def jax_zero_external_loads(self, state, time):
        updated = dict(state)
        updated["external_force"] = jnp.asarray(0.0)
        return updated


class FakeJAXCollection:
    def __init__(self, *systems):
        self._systems = systems

    def final_systems(self):
        yield from self._systems

    def jax_constrain_values(self, states, time):
        return tuple(
            {**state, "position": state["position"] + jnp.asarray(10.0)}
            for state in states
        )

    def jax_synchronize(self, states, time):
        return tuple(
            {**state, "acceleration": state["internal_force"] + jnp.asarray(1.0)}
            for state in states
        )

    def jax_constrain_rates(self, states, time):
        return tuple(
            {**state, "velocity": state["velocity"] + jnp.asarray(3.0)}
            for state in states
        )


def test_position_verlet_gpu_integrate_uses_fori_loop_protocol():
    system = FakeJAXBlock()
    collection = FakeJAXCollection(system)
    stepper = PositionVerletGPU()

    with jax.default_device(jax.devices("cpu")[0]):
        final_time = stepper.integrate(
            collection,
            time=np.float64(0.0),
            final_time=np.float64(0.2),
            dt=np.float64(0.2),
        )

    np.testing.assert_allclose(final_time, np.float64(0.2))
    np.testing.assert_allclose(np.asarray(system.final_state["position"]), 20.522)
    np.testing.assert_allclose(np.asarray(system.final_state["velocity"]), 4.22)
    np.testing.assert_allclose(np.asarray(system.final_state["acceleration"]), 1.1)
    np.testing.assert_allclose(np.asarray(system.final_state["internal_force"]), 0.1)
    np.testing.assert_allclose(np.asarray(system.final_state["external_force"]), 0.0)


def test_position_verlet_gpu_integrate_handles_multiple_systems():
    system_one = FakeJAXBlock()
    system_two = FakeJAXBlock()
    collection = FakeJAXCollection(system_one, system_two)
    stepper = PositionVerletGPU()

    with jax.default_device(jax.devices("cpu")[0]):
        final_time = stepper.integrate(
            collection,
            time=np.float64(0.0),
            final_time=np.float64(0.4),
            dt=np.float64(0.2),
        )

    np.testing.assert_allclose(final_time, np.float64(0.4))
    np.testing.assert_allclose(np.asarray(system_one.final_state["position"]), 41.692)
    np.testing.assert_allclose(np.asarray(system_two.final_state["position"]), 41.692)


def test_position_verlet_gpu_integrate_rejects_nonpositive_dt():
    stepper = PositionVerletGPU()
    collection = FakeJAXCollection(FakeJAXBlock())

    with pytest.raises(AssertionError):
        stepper.integrate(
            collection,
            time=np.float64(0.0),
            final_time=np.float64(0.2),
            dt=np.float64(0.0),
        )


def test_position_verlet_gpu_integrate_rejects_inconsistent_final_time():
    stepper = PositionVerletGPU()
    collection = FakeJAXCollection(FakeJAXBlock())

    with pytest.raises(AssertionError):
        stepper.integrate(
            collection,
            time=np.float64(0.0),
            final_time=np.float64(0.25),
            dt=np.float64(0.1),
        )
