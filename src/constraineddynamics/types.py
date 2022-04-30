from typing import NewType

from numpy import ndarray

SimVec = NewType("SimVec", ndarray) # length 2 * n_particles * dim
QVec = NewType("QVec", ndarray) # length n_particles * dim
SpaceVec = NewType("SpaceVec", ndarray) # length dim
ParticleVec = NewType("ParticleVec", ndarray) # length n_particles
ConstraintVec = NewType("ConstraintVec", ndarray) # length len(constraints)

SpaceVecs = NewType("SpaceVecs", ndarray) # n_particles x dim
QMatrix = NewType("QMatrix", ndarray) # (n_particles * dim) x (n_particles * dim)