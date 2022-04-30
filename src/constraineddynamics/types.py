from typing import NewType

from numpy import ndarray

SimVec = NewType("SimVec", ndarray)
SpaceVec = NewType("SpaceVec", ndarray)
SpaceVecs = NewType("SpaceVecs", ndarray)
QVec = NewType("QVec", ndarray)
ParticleVec = NewType("ParticleVec", ndarray)

QMatrix = NewType("QMatrix", ndarray)