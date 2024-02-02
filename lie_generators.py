import numpy as np

_generators = {
    (2, "fundamental"): np.array([
        [[0, 1], [1, 0]],
        [[0, -1j], [1j, 0]],
        [[1, 0], [0, -1]]]) / 2.,
    (3, "fundamental"): np.array([
        [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
        [[0, 0, 1], [0, 0, 0], [1, 0, 0]],
        [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
        [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]],
        [[0, 0, -1j], [0, 0, 0], [1j, 0, 0]],
        [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]],
        [[1, 0, 0], [0, -1, 0], [0, 0, 0]],
        [[1. / (np.sqrt(3.)), 0, 0], [0, 1. / (np.sqrt(3.)), 0], [0, 0, -2. / (np.sqrt(3.))]]]) / 2.
}


def generators(nc, representation="fundamental"):
    if (nc, representation) in _generators:
        return _generators[(nc, representation)]
    if representation == "adjoint":
        fundamental_lie_generators = generators(nc, representation="fundamental")
        adjoint_generators = np.zeros(shape=(nc * nc - 1, nc * nc - 1, nc * nc - 1), dtype=np.complex128)
        for i in range(nc * nc - 1):
            for j in range(nc * nc - 1):
                for k in range(nc * nc - 1):
                    adjoint_generators[i, j, k] = -2 * np.trace((fundamental_lie_generators[i] @
                                                                 fundamental_lie_generators[j] -
                                                                 fundamental_lie_generators[j] @
                                                                 fundamental_lie_generators[i]) @
                                                                fundamental_lie_generators[k])

        _generators[(nc, representation)] = adjoint_generators

        return adjoint_generators
