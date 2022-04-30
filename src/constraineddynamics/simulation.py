from typing import List

import numpy as np
import scipy.linalg as lin
import scipy.integrate as nint

from .constraint import ConstraintMapper, Constraint
from .particle_system import ParticleSystem


class Simulator:
    def __init__(self, particle_system: ParticleSystem, constraints: List[Constraint]):
        self.particle_system = particle_system
        self.constraint_mapper = ConstraintMapper(constraints=constraints, ps=particle_system)

    def lagrange_multiplier(self, q, qdot, f):
        J = self.constraint_mapper.constraint_jac(q)
        W = self.particle_system.inverse_mass(q)
        JW = np.dot(J, W)
        A = np.dot(JW, J.T)
        b = (
                -np.dot(self.constraint_mapper.constraint_jac_dt(q, qdot), qdot)
                - np.dot(JW, f)
                - 1.0 * self.constraint_mapper.constraint(q) - 1.0 * self.constraint_mapper.constraint_dt(q, qdot)
        )
        return lin.lstsq(A, b)[0]

    def net_acceleration(self, q, qdot, f):
        J = self.constraint_mapper.constraint_jac(q)
        W = self.particle_system.inverse_mass(q)
        f_constraint = np.dot(J.T, self.lagrange_multiplier(q, qdot, f))
        return np.dot(W, f + f_constraint)

    def _ode_y(self, t, y):
        x, xdot = self.particle_system.y_to_qqdot(y)
        dx = xdot
        dxdot = self.net_acceleration(x, xdot, self.force(x))
        return self.particle_system.qqdot_to_y(dx, dxdot)

    def simulate(self, y0, t_range):
        return nint.solve_ivp(self._ode_y,
                              y0=y0,
                              t_span=(t_range[0], t_range[-1]),
                              t_eval=t_range,
                              dense_output=True
                              )

    def force(self, q):
        return np.kron(self.particle_system.masses, -np.eye(len(q) // self.particle_system.n_particles)[1])
