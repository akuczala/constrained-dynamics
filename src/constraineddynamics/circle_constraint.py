from dataclasses import dataclass

import numpy as np

from .constraint import MultipleParticleConstraint, SingleParticleConstraint


@dataclass
class CircleConstraint(SingleParticleConstraint):
    origin: np.ndarray
    radius: float

    def delta_x(self, x):
        return x - self.origin

    def constraint(self, q) -> float:
        return self.apply_to_particle(self.particle_constraint, q)

    def particle_constraint(self, x) -> float:
        return 0.5 * ((lambda xr: np.dot(xr, xr))(self.delta_x(x)) - self.radius ** 2)

    def constraint_jac(self, q):
        return self.apply_to_particle_vec(self.delta_x, q)

    def constraint_jac_dt(self, q, qdot):
        return self.apply_to_particle_vec(lambda x: x, qdot)


@dataclass
class CoupledCircleConstraint(MultipleParticleConstraint):
    """
    Particle 0 sets the origin of the circle
    Particle 1 is constrained to the circle
    """
    radius: float

    def delta_x(self, xs):
        return self.on_circle_x(xs) - self.origin(xs)

    def origin(self, xs):
        return xs[0]

    def on_circle_x(self, xs):
        return xs[1]

    def constraint(self, q) -> float:
        return self.apply_to_particles(self.circle_constraint, q)

    def circle_constraint(self, xs) -> float:
        return 0.5 * ((lambda xr: np.dot(xr, xr))(self.delta_x(xs)) - self.radius ** 2)

    def constraint_jac(self, q):
        # print(self.delta_x(self.ps.q_to_xs(q)))
        return self.apply_to_particles_vec([lambda xs: -self.delta_x(xs), self.delta_x], q)

    def constraint_jac_dt(self, q, qdot):
        return self.apply_to_particles_vec([lambda xdots: -self.delta_x(xdots), self.delta_x], qdot)
