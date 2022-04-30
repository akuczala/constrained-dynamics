from dataclasses import dataclass

import numpy as np


@dataclass
class ParticleSystem:
    dim: int
    n_particles: int

    def y_to_q(self, y):
        return y[:len(y) // 2]

    def y_to_qdot(self, y):
        return y[len(y) // 2:]

    def y_to_qqdot(self, y):
        return self.y_to_q(y), self.y_to_qdot(y)

    def qqdot_to_y(self, x, xdot):
        return np.hstack([x, xdot])

    def q_to_xs(self, q):
        return q.reshape(-1, self.dim)

    def xs_to_q(self, xs):
        return xs.ravel()