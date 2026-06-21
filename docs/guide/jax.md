# JAX Integration Guide

This page documents the current JAX integration in PyElastica as it exists today.

The JAX path is intended for device-backed simulation rollouts where the block state
stays on the accelerator during time stepping, and host readback happens explicitly
only when needed.

## Current Scope

The current JAX integration provides:

- `MemoryBlockCosseratRodJax` as a device-backed block implementation for Cosserat rods.
- `PositionVerletGPU` as the current JAX-owned time stepper.
- `JAXCompatible` for simulator collections that need JAX-stage execution hooks.
- `JAXOps` for registering pure-JAX rod-local operators that are lowered at `finalize()`.
- `NoOpsJax` and example JAX operations such as `GravityAnalyticalDamperJax`.

The intended execution model is:

1. Create rods as usual.
2. Build a simulator with `BaseSystemCollection` and the JAX mixins.
3. Register JAX operators with `simulator.using(rod).operate(...)`.
4. Call `finalize()`, which packs rods into `MemoryBlockCosseratRodJax` and lowers
   the rod-local JAX operators onto block state.
5. Run `PositionVerletGPU.integrate(...)`.

## What Is Different From The CPU Path

The CPU path in PyElastica is based on mutable live objects. Existing modules such as
forcing, damping, and constraints typically:

- receive a `system` object
- mutate `system.position_collection`, `system.external_forces`, and similar arrays
- rely on shared memory layout after block construction

The JAX path does not work that way.

JAX requires:

- explicit state passed through the time-stepping loop
- pure staged transforms
- no hidden Python-side mutation during traced execution

Because of that, the current JAX path does **not** directly reuse the existing host
module implementations at runtime.

## Current Limitation: Not All Mixins Are Supported

The current JAX integration does **not** support all existing simulator mixins.

At the moment, the JAX path is structured around:

- `BaseSystemCollection`
- `JAXCompatible`
- `JAXOps`

Existing mixins such as:

- `Forcing`
- `Damping`
- `Constraints`
- `Connections`
- `Contact`
- `CallBacks`

are still part of the standard CPU path, but their existing host-side operator classes
are not automatically reusable inside the pure JAX rollout.

This is a deliberate restriction of the current implementation, not a bug.

## Current Limitation: Load Classes Must Be Re-implemented For JAX

If you want a load or operator to participate in pure JAX rollout, it currently needs
to be implemented as a JAX operator class.

In other words:

- existing host-side load classes are still valid for CPU simulations
- they are **not** automatically lowered into JAX
- a JAX version must be written in the stateful rod-view style

The current JAX operator syntax is:

```python
import elastica as ea


class MyJAXLoad(ea.NoOpsJax):
    def __init__(self, magnitude, *, _system=None):
        self.magnitude = magnitude

    def jax_operate_synchronize(self, rod_view, time):
        rod_view.external_forces = (
            rod_view.external_forces + self.magnitude * rod_view.mass[None, :]
        )
        return rod_view
```

and it is registered as:

```python
simulator.using(rod).operate(MyJAXLoad, magnitude=2.0)
```

## The Rod View Contract

JAX operators do not receive the original rod object during rollout.

Instead, at `finalize()`, each registered JAX operator is lowered against a rod-local
view into the packed JAX block state. The operator receives a `rod_view` object that
supports attribute-style access:

```python
rod_view.position_collection
rod_view.velocity_collection
rod_view.external_forces
rod_view.mass
```

and whole-field replacement:

```python
rod_view.external_forces = new_external_forces
return rod_view
```

This keeps the syntax close to existing PyElastica code while preserving the explicit
state model required by JAX.

:::{important}
The current rod-view design is intended for whole-field replacement, not arbitrary
NumPy-style in-place mutation of indexed subviews. Prefer:

```python
forces = rod_view.external_forces
forces = forces.at[..., -1].add(tip_force)
rod_view.external_forces = forces
```

over relying on Python-side in-place mutation.
:::

## Example Simulator Setup

```python
import elastica as ea


class JAXSimulator(
    ea.BaseSystemCollection,
    ea.JAXCompatible,
    ea.JAXOps,
):
    pass
```

Registering a rod-local JAX operator:

```python
simulator = JAXSimulator()
simulator.enable_block_supports(ea.CosseratRod, ea.MemoryBlockCosseratRodJax)

rod = ea.CosseratRod.straight_rod(...)
simulator.append(rod)
simulator.using(rod).operate(MyJAXLoad, magnitude=2.0)
simulator.finalize()
```

Running the JAX stepper:

```python
stepper = ea.PositionVerletGPU()
stepper.integrate(
    simulator,
    time=0.0,
    final_time=1.0e-3,
    dt=1.0e-5,
)
```

## Example Built-in JAX Operator

`GravityAnalyticalDamperJax` is included as an example of the current JAX operator style.

It combines:

- gravity loading in `jax_operate_synchronize`
- analytical rate damping in `jax_operate_constrain_rates`

It is registered in the same way:

```python
simulator.using(rod).operate(
    ea.GravityAnalyticalDamperJax,
    acc_gravity=np.array([0.0, -9.80665, 0.0]),
    uniform_damping_constant=5.0,
    time_step=1.0e-5,
)
```

## Recommended Mental Model

For the current implementation, it is best to think of the JAX path as a parallel
execution path with its own operator classes, not as a drop-in acceleration of every
existing host-side module.

Use this rule:

- CPU path: existing mixins and existing load classes
- JAX path: `JAXCompatible + JAXOps` and JAX-specific operator classes

This keeps the existing API intact while allowing pure device-side rollout where
supported.
