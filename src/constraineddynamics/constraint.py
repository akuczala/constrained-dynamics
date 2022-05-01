from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import List, TypeVar, Callable, NewType

import numpy as np

from .particle_system import ParticleSystem
from .types import QVec, SpaceVecs, SpaceVec

_T = TypeVar("_T")


@dataclass(frozen=True)
class Constraint(ABC):
    ps: ParticleSystem

    @abstractmethod
    def constraint(self, q: QVec):
        pass

    def constraint_dt(self, q: QVec, qdot: QVec):
        return np.dot(self.constraint_jac(q), qdot)

    @abstractmethod
    def constraint_jac(self, q: QVec):
        pass

    @abstractmethod
    def constraint_jac_dt(self, q: QVec, qdot: QVec):
        pass


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class SingleParticleConstraint(Constraint, ABC):
    apply_to: int

    def apply_to_particle(self, f: Callable[[SpaceVec], _T], q: QVec) -> _T:
        return f(SpaceVec(self.ps.q_to_xs(q)[self.apply_to]))

    def apply_to_particle_vec(self, f: Callable[[SpaceVec], SpaceVec], q: QVec) -> QVec:
        return self.ps.xs_to_q(SpaceVecs(np.array([
            f(x) if i == self.apply_to else np.zeros_like(x)
            for i, x in enumerate(self.ps.q_to_xs(q))
        ])))


ApplicableSpaceVecs = NewType("ApplicableSpaceVecs", np.ndarray)


@dataclass(frozen=True)
class MultipleParticleConstraint(Constraint, ABC):
    apply_to: List[int]

    def get_particles(self, q: QVec) -> ApplicableSpaceVecs:
        return self.ps.q_to_xs(q)[self.apply_to]

    def apply_to_particles(self, f: Callable[[ApplicableSpaceVecs], _T], q: QVec) -> _T:
        return f(self.get_particles(q))

    def apply_to_particles_vec(self, fs: List[Callable[[ApplicableSpaceVecs], SpaceVec]], q: QVec) -> QVec:
        f_iter = iter(fs)
        xs = self.get_particles(q)
        return self.ps.xs_to_q(SpaceVecs(np.array([
            next(f_iter)(xs) if i in self.apply_to else np.zeros([self.ps.dim])
            for i in range(self.ps.n_particles)
        ])))
