import numpy as np

from src.constraineddynamics.circle_constraint import CircleConstraint, CoupledCircleConstraint
from src.constraineddynamics.particle_system import ParticleSystem
from src.constraineddynamics.simulation import Simulator


def run_simulator_methods(simulator: Simulator, q0, qdot0):
    print(
        simulator.lagrange_multiplier(q0, qdot0, simulator.force(q0))
    )
    print(
        simulator.net_acceleration(q0, qdot0, simulator.force(q0))
    )


def test_single_particle():
    particle_system = ParticleSystem(dim=2, masses=np.array([1.0]))
    constraints = [CircleConstraint(origin=np.array([0.0, 0.0]), radius=1, apply_to=0, ps=particle_system)]
    simulator = Simulator(particle_system, constraints)

    q0 = particle_system.xs_to_q(np.array([
        [1, 0.0]
    ]))
    qdot0 = particle_system.xs_to_q(np.array([
        [0, 1.0]
    ]))

    run_simulator_methods(simulator, q0, qdot0)


def test_decoupled_particles():
    particle_system = ParticleSystem(dim=2, masses=np.array([1.0, 1.0]))
    constraints = [
        CircleConstraint(origin=np.array([0.0, 0.0]), radius=1, apply_to=0, ps=particle_system),
        CircleConstraint(origin=np.array([1.0, 0.0]), radius=3, apply_to=1, ps=particle_system)
    ]
    simulator = Simulator(particle_system, constraints)

    q0 = particle_system.xs_to_q(np.array([
        [1, 0.0],
        [4, 0.0]
    ]))
    qdot0 = particle_system.xs_to_q(np.array([
        [0, 1.0],
        [0, 1.0]
    ]))

    run_simulator_methods(simulator, q0, qdot0)


def test_coupled_particles():
    particle_system = ParticleSystem(dim=2, masses=np.array([1.0, 1.0]))
    constraints = [
        CircleConstraint(origin=np.array([0.0, 0.0]), radius=3, apply_to=0, ps=particle_system),
        CoupledCircleConstraint(radius=1, apply_to=[0, 1], ps=particle_system),
    ]
    simulator = Simulator(particle_system, constraints)

    q0 = particle_system.xs_to_q(np.array([
        [3, 0.0],
        [4, 0.0]
    ]))
    qdot0 = particle_system.xs_to_q(np.array([
        [0, 0.0],
        [0, 0.0]
    ]))

    run_simulator_methods(simulator, q0, qdot0)


def test_multiple_coupled_particles():
    particle_system = ParticleSystem(dim=2, masses=np.array([1.0, 1.0, 1.0, 1.0]))
    constraints = [
        CircleConstraint(origin=np.array([0.0, 0.0]), radius=3, apply_to=0, ps=particle_system),
        CoupledCircleConstraint(radius=1, apply_to=[0, 1], ps=particle_system),
        CoupledCircleConstraint(radius=1, apply_to=[1, 2], ps=particle_system),
        CoupledCircleConstraint(radius=1, apply_to=[2, 3], ps=particle_system)
    ]
    simulator = Simulator(particle_system, constraints)

    q0 = particle_system.xs_to_q(np.array([
        [3, 0.0],
        [4, 0.0],
        [5, 0.0],
        [6, 0.0],
    ]))
    qdot0 = particle_system.xs_to_q(np.array([
        [0, 0.0],
        [0, 0.0],
        [0, 0.0],
        [0, 0.0],
    ]));

    run_simulator_methods(simulator, q0, qdot0)
