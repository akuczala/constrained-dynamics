from typing import List

import numpy as np
import scipy.linalg as lin
import scipy.integrate as nint

from .constraint import ConstraintMapper, Constraint
from .particle_system import ParticleSystem


def inverse_mass(q, m):
    return np.diag(np.repeat(1 / m, len(q) // len(m)))


def force(q, m):
    return np.kron(m, -np.eye(len(q) // len(m))[1])


def lagrange_multiplier(constraint_mapper: ConstraintMapper, q, qdot, f, m):
    J = constraint_mapper.constraint_jac(q)
    W = inverse_mass(q, m)
    JW = np.dot(J, W)
    A = np.dot(JW, J.T)
    b = (
            -np.dot(constraint_mapper.constraint_jac_dt(q, qdot), qdot)
            - np.dot(JW, f)
            - 1.0 * constraint_mapper.constraint(q) - 1.0 * constraint_mapper.constraint_dt(q, qdot)
    )
    return lin.lstsq(A, b)[0]


def net_acceleration(constraint_mapper: ConstraintMapper, q, qdot, f, m):
    J = constraint_mapper.constraint_jac(q)
    W = inverse_mass(q, m)
    f_constraint = np.dot(J.T, lagrange_multiplier(constraint_mapper, q, qdot, f, m))
    return np.dot(W, f + f_constraint)


def ode_y(t, y, particle_system: ParticleSystem, constraint_mapper: ConstraintMapper, m):
    x, xdot = particle_system.y_to_qqdot(y)
    dx = xdot
    dxdot = net_acceleration(constraint_mapper, x, xdot, force(x, m), m=m)
    return particle_system.qqdot_to_y(dx, dxdot)


def simulate(particle_system: ParticleSystem, constraints: List[Constraint], masses, y0, t_range):
    return nint.solve_ivp(ode_y,
                         y0=y0,
                         t_span=(t_range[0], t_range[-1]),
                         t_eval=t_range,
                         args=(particle_system, ConstraintMapper(constraints=constraints, ps=particle_system), masses),
                         dense_output=True
                         )
