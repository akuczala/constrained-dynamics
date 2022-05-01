from dataclasses import dataclass

import numpy as np

from .constraint import SingleParticleConstraint
from .types import QVec, SpaceVec


@dataclass(frozen=True)
class PlanarConstraint(SingleParticleConstraint):
    n: SpaceVec
    p: SpaceVec

    def constraint(self, q: QVec):
        return self.apply_to_particle(self.planar_constraint, q)

    def planar_constraint(self, x: SpaceVec) -> float:
        return float(np.dot(self.n, x - self.p))

    def constraint_jac(self, q: QVec):
        return self.apply_to_particle_vec(lambda x: self.n, q)

    def constraint_jac_dt(self, q: QVec, qdot: QVec):
        return self.apply_to_particle_vec(lambda x: SpaceVec(np.zeros_like(x)), q)
