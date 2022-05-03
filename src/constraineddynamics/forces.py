from typing import Callable

import numpy as np

from constraineddynamics.types import ParticleVec, QVec


def get_surface_gravity_force(g: float, masses: ParticleVec) -> Callable[[QVec, QVec], QVec]:
    def force(q: QVec, qdot: QVec) -> QVec:
        return QVec(
            np.kron(g * masses, -np.eye(len(q) // len(masses))[1])
        )

    return force
