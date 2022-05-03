from dataclasses import dataclass
from typing import Tuple, Callable

import numpy as np

from .types import ParticleVec, SimVec, QVec, SpaceVecs, QMatrix


@dataclass
class ParticleSystem:
    dim: int
    masses: ParticleVec
    force: Callable[[QVec, QVec], QVec]

    @property
    def n_particles(self) -> int:
        return len(self.masses)

    def y_to_q(self, y: SimVec) -> QVec:
        return QVec(y[:len(y) // 2])

    def y_to_qdot(self, y: SimVec) -> QVec:
        return QVec(y[len(y) // 2:])

    def y_to_qqdot(self, y: SimVec) -> Tuple[QVec, QVec]:
        return self.y_to_q(y), self.y_to_qdot(y)

    def qqdot_to_y(self, x: QVec, xdot: QVec) -> SimVec:
        return SimVec(np.hstack([x, xdot]))

    def q_to_xs(self, q: QVec) -> SpaceVecs:
        return SpaceVecs(q.reshape(-1, self.dim))

    def xs_to_q(self, xs: SpaceVecs) -> QVec:
        return QVec(xs.ravel())

    def inverse_mass(self, q: QVec) -> QMatrix:
        return QMatrix(np.diag(np.repeat(1 / self.masses, self.dim)))
