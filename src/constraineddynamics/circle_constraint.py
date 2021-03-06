from dataclasses import dataclass

import numpy as np

from .constraint import MultipleParticleConstraint, SingleParticleConstraint, ApplicableSpaceVecs
from .types import SpaceVec, QVec


@dataclass(frozen=True)
class CircleConstraint(SingleParticleConstraint):
    origin: SpaceVec
    radius: float

    def delta_x(self, x: SpaceVec) -> SpaceVec:
        return x - self.origin

    def particle_constraint(self, x: SpaceVec) -> float:
        return 0.5 * ((lambda xr: np.dot(xr, xr))(self.delta_x(x)) - self.radius ** 2)

    def constraint(self, q: QVec):
        return self.apply_to_particle(self.particle_constraint, q)

    def constraint_jac(self, q: QVec):
        return self.apply_to_particle_vec(self.delta_x, q)

    def constraint_jac_dt(self, q: QVec, qdot: QVec):
        return self.apply_to_particle_vec(lambda x: x, qdot)


@dataclass(frozen=True)
class CoupledCircleConstraint(MultipleParticleConstraint):
    """
    Particle 0 sets the origin of the circle
    Particle 1 is constrained to the circle
    """
    radius: float

    def delta_x(self, xs: ApplicableSpaceVecs) -> SpaceVec:
        return self.on_circle_x(xs) - self.origin(xs)

    def origin(self, xs: ApplicableSpaceVecs) -> SpaceVec:
        return xs[0]

    def on_circle_x(self, xs: ApplicableSpaceVecs) -> SpaceVec:
        return xs[1]

    def constraint(self, q: QVec) -> float:
        return self.apply_to_particles(self.circle_constraint, q)

    def circle_constraint(self, xs: ApplicableSpaceVecs) -> float:
        return 0.5 * ((lambda xr: np.dot(xr, xr))(self.delta_x(xs)) - self.radius ** 2)

    def constraint_jac(self, q: QVec):
        return self.apply_to_particles_vec([lambda xs: SpaceVec(-self.delta_x(xs)), self.delta_x], q)

    def constraint_jac_dt(self, q, qdot: QVec):
        return self.apply_to_particles_vec([lambda xdots: SpaceVec(-self.delta_x(xdots)), self.delta_x], qdot)
