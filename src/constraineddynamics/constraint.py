from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import List

import numpy as np

from .particle_system import ParticleSystem


@dataclass
class Constraint(ABC):
    ps: ParticleSystem

    @abstractmethod
    def constraint(self, q):
        pass

    def constraint_dt(self, q, qdot):
        return np.dot(self.constraint_jac(q), qdot)

    @abstractmethod
    def constraint_jac(self, q):
        pass

    @abstractmethod
    def constraint_jac_dt(self, q, qdot):
        pass


@dataclass
class ConstraintMapper(Constraint):
    constraints: List[Constraint]

    def map_constraints(self, f, q, qdot=None):
        return np.array([f(c)(q) if qdot is None else f(c)(q, qdot) for c in self.constraints])

    def constraint(self, q):
        return self.map_constraints(lambda c: c.constraint, q)

    def constraint_jac(self, q):
        return self.map_constraints(lambda c: c.constraint_jac, q)

    def constraint_jac_dt(self, q, qdot):
        return self.map_constraints(lambda c: c.constraint_jac_dt, q, qdot)


@dataclass
class SingleParticleConstraint(Constraint, ABC):
    apply_to: int

    def apply_to_particle(self, f, q):
        return f(self.ps.q_to_xs(q)[self.apply_to])

    def apply_to_particle_vec(self, f, q):
        return self.ps.xs_to_q(np.array([
            f(x) if i == self.apply_to else np.zeros_like(x)
            for i, x in enumerate(self.ps.q_to_xs(q))
        ]))


@dataclass
class MultipleParticleConstraint(Constraint, ABC):
    apply_to: List[int]

    def get_particles(self, q):
        return self.ps.q_to_xs(q)[self.apply_to]

    def apply_to_particles(self, f, q):
        return f(self.get_particles(q))

    def apply_to_particles_vec(self, fs, q):
        f_iter = iter(fs)
        xs = self.get_particles(q)
        return self.ps.xs_to_q(np.array([
            next(f_iter)(xs) if i in self.apply_to else np.zeros([self.ps.dim])
            for i in range(self.ps.n_particles)
        ]))
