import tensorflow as tf
from translate import translate, lookup_tables, global_sum
import lattice as lt
from measurements import plaquette
from smearing import unitary_sqrt_projection
import lie_generators
from metropolis import metropolis
import copy


# ************************************************
# In the beginning, gauge fields are initialized randomly
# (hotstart).
# ************************************************

@tf.function
def random_gauge_field(nc=3, nd=4):
    '''Return a random gauge field with nc colors'''
    volume = lookup_tables.local_volume
    random_field = tf.complex(
        real=tf.random.normal(shape=(nd, nc, nc, volume), dtype=tf.float64),
        imag=tf.random.normal(shape=(nd, nc, nc, volume), dtype=tf.float64))

    return unitary_sqrt_projection(random_field)


def hotstart(geometry, nc=3, nd=4):
    return lt.Configuration(gauge_field=random_gauge_field(nc, nd),
                            geometry=geometry,
                            colors=nc,
                            number_of_dimensions=nd)


# ************************************************
# The link field is updated at each integration step
# exponentiating the momenta (Lie algebra variables) to get
# SU(N) matrix (Lie group variables)
# ************************************************

@tf.function
def update_link_field(link_field, momenta, epsilon):
    exp = tf.linalg.expm(tf.transpose(epsilon * momenta, [3, 0, 1, 2]))
    return tf.einsum("mijc,mjkc->mikc", tf.transpose(exp, [1, 2, 3, 0]), link_field)


def update_configuration(config: lt.Configuration, momenta, epsilon):
    return lt.Configuration(
        gauge_field=update_link_field(config.gauge_field, momenta, epsilon),
        geometry=config.geometry,
        colors=config.colors,
        number_of_dimensions=config.number_of_dimensions)


# ************************************************
# The momenta are updated with the gauge force (Hamilton
# equations).
# ************************************************

@tf.function
def update_momenta(momenta, gauge_force, epsilon):
    return momenta + epsilon * gauge_force


# ************************************************
# The momenta are initialized accordingly to a Gaussian
# distribution (the quadratic term of the Hamiltonian).
# ************************************************

@tf.function
def random_momenta(nc=3, nd=4):
    volume = lookup_tables.local_volume

    momenta = [[[0 for _ in range(nc)] for _ in range(nc)] for _ in range(nd)]
    for mu in range(nd):
        for i in range(nc):
            for j in range(i + 1, nc):
                random_field_real = tf.random.normal(shape=(volume,), stddev=1., dtype=tf.float64) / 2.
                random_field_imag = tf.random.normal(shape=(volume,), stddev=1., dtype=tf.float64) / 2.

                momenta[mu][i][j] = tf.complex(random_field_real, random_field_imag)
                momenta[mu][j][i] = tf.complex(-random_field_real, random_field_imag)

            momenta[mu][i][i] = []

        for i in range(1, nc):
            random_field_imag = tf.random.normal(shape=(volume,), stddev=1., dtype=tf.float64) / 2.
            ii = tf.constant(i, dtype=tf.float64)
            for j in range(i):
                momenta[mu][j][j].append(random_field_imag / tf.math.sqrt(ii * (ii + 1) / 2.))
            momenta[mu][i][i].append(-random_field_imag * (ii * tf.math.sqrt(2. / (ii * (ii + 1)))))

        for i in range(nc):
            momenta[mu][i][i] = tf.complex(real=tf.zeros_like(random_field_imag),
                                           imag=tf.math.add_n(momenta[mu][i][i]))

    return tf.convert_to_tensor(momenta, dtype=tf.complex128)


# ************************************************
# Hybrid Monte Carlo requires a reversible numerical
# integrator, such as a quadratic leap frog
# ************************************************

def integrate_lp(config, momenta, actions, steps, delta_t, action_index=0):
    '''Leap-frog reversible numerical integrator'''
    epsilon = delta_t / steps[action_index]
    print("Integrating", action_index, "with step", epsilon)

    actions[action_index].set_gauge_configuration(config)
    momenta = update_momenta(momenta, actions[action_index].force(), - (epsilon / 2))

    for _ in range(steps[action_index]):
        if action_index == len(actions) - 1:
            config = update_configuration(config, momenta, epsilon)
        else:
            config, momenta = integrate_lp(config, momenta, actions, steps, epsilon, action_index + 1)

        actions[action_index].set_gauge_configuration(config)
        force = actions[action_index].force()
        momenta = update_momenta(momenta, force, - epsilon)

    momenta = update_momenta(momenta, force, + epsilon / 2)

    return config, momenta


# ************************************************
# The Omelyan integrator is a reversible second order
# integrator, which is tuned to reduce to the the next order
# integration error.
# ************************************************

def integrate_om(config, momenta, actions, steps, delta_t, action_index=0):
    '''Omelyan second-order integrator, ll is a parameter tuned to minimize energy violations'''
    ll = 0.19318332275037863
    epsilon = delta_t / steps[action_index]

    actions[action_index].set_gauge_configuration(config)
    momenta = update_momenta(momenta, actions[action_index].force(), - ll * epsilon)

    for _ in range(steps[action_index]):
        if action_index == len(actions) - 1:
            config = update_configuration(config, momenta, epsilon / 2)
        else:
            config, momenta = integrate_om(config, momenta, actions, steps, epsilon / 2, action_index + 1)

        actions[action_index].set_gauge_configuration(config)
        force = actions[action_index].force()
        momenta = update_momenta(momenta, force, -(1 - 2 * ll) * epsilon)

        if action_index == len(actions) - 1:
            config = update_configuration(config, momenta, epsilon / 2)
        else:
            config, momenta = integrate_om(config, momenta, actions, steps, epsilon / 2, action_index + 1)

        actions[action_index].set_gauge_configuration(config)
        force = actions[action_index].force()
        momenta = update_momenta(momenta, force, -2 * ll * epsilon)

    momenta = update_momenta(momenta, force, +ll * epsilon)

    return config, momenta


# ************************************************
# The energy of the momenta is just the trace of the square.
# The Hamiltonian is quadratic in the momenta.
# ************************************************

@tf.function
def trace_momenta(momenta):
    return -tf.math.real(tf.reduce_sum(tf.einsum("mijc,mjic->mc", momenta, momenta)))


def momenta_energy(momenta):
    return global_sum(trace_momenta(momenta))


# ************************************************
# The gauge energy is just equal to the plaquette times the
# inverse gauge coupling squared (=beta), up to a volume
# factor.
# ************************************************

def gauge_energy(cfg: lt.Configuration, beta):
    volume = lookup_tables.global_volume

    factor = beta * (cfg.number_of_dimensions - 1) * (cfg.number_of_dimensions) / 2

    return -factor * volume * plaquette(cfg).value


# Dirac-Wilson operators


def generate_random_vector(nc: int = 3, spinor_dimension: int = 4, representation: str = "fundamental",
                           stddev: float = 1):
    volume = lookup_tables.local_volume
    if representation == "fundamental":
        return tf.complex(
            real=tf.random.normal(shape=(spinor_dimension, nc, volume), dtype=tf.float64, stddev=stddev),
            imag=tf.random.normal(shape=(spinor_dimension, nc, volume), dtype=tf.float64, stddev=stddev))
    elif representation == "adjoint":
        generators = lie_generators.generators(nc, representation="fundamental")

        vector = tf.complex(
            real=tf.random.normal(shape=(spinor_dimension, nc * nc - 1, volume), dtype=tf.float64, stddev=stddev),
            imag=tf.random.normal(shape=(spinor_dimension, nc * nc - 1, volume), dtype=tf.float64, stddev=stddev))

        return tf.einsum("mic,ijk->mjkc", vector, generators)


# ************************************************
# Hybrid Monte Carlo, integrating the equation of motion for
# n steps for a trajectory of length delta_t.
# ************************************************

def hybrid_mc(config: lt.Configuration, actions, steps, delta_t):
    # random initilization of the momenta
    momenta = random_momenta(config.colors, config.number_of_dimensions)
    # Compute the energy at the beginning of the trajectory
    old_energy = momenta_energy(momenta)
    for action in actions:
        action.set_gauge_configuration(config)
        old_energy += action.energy()

    new_config, new_momenta = integrate_om(
        config,
        momenta,
        actions,
        steps,
        delta_t)

    # Compute the energy at the end of the trajectory
    new_energy = momenta_energy(new_momenta)
    for action in actions:
        action.set_gauge_configuration(new_config)
        new_energy += action.energy()

    # Accept/reject step
    if metropolis(new_energy, old_energy):
        print("Delta energy in Metropolis", (new_energy - old_energy), "accepted!")
        return new_config, -(new_energy - old_energy), True
    else:
        print("Delta energy in Metropolis", (new_energy - old_energy), "rejected!")
        return copy.deepcopy(config), -(new_energy - old_energy), False


def reversibility_check(config: lt.Configuration, actions, steps, delta_t):
    # random initilization of the momenta
    momenta = random_momenta(config.colors, config.number_of_dimensions)
    # Compute the energy at the beginning of the trajectory
    old_energy = momenta_energy(momenta)
    for action in actions:
        action.set_gauge_configuration(config)
        old_energy += action.energy()

    new_config, new_momenta = integrate_om(
        config,
        momenta,
        actions,
        steps,
        delta_t)

    new_config, new_momenta = integrate_om(
        new_config,
        new_momenta,
        actions,
        steps,
        -delta_t)

    # Compute the energy at the end of the trajectory
    new_energy = momenta_energy(new_momenta)
    for action in actions:
        action.set_gauge_configuration(new_config)
        new_energy += action.energy()

    return new_energy - old_energy

