from collections import namedtuple
import tensorflow as tf
import lattice as lt
from translate import translate, lookup_tables, global_sum
import numpy as np

Measurement = namedtuple("Measurement", ["geometry", "colors", "value", "name", "smearing_levels"])
Measurement.__new__.__defaults__ = (None, None, None, None, 0)

# ************************************************
# The plaquette corresponds to the trace of the square of the
# field strength tensor F_\mu\nu^ab , and it is the basic
# ingredient to define the gauge action
# ************************************************

@tf.function
def trace_plaquette(links1, links2, links3, links4):
    '''Compute the trace of the links in one plaquette'''
    return tf.einsum("klc,lmc,nmc,knc", links1, links2, tf.math.conj(links3), tf.math.conj(links4))


def plaquette(cfg: lt.Configuration) -> Measurement:
    """Return the plaquette expectation value"""
    # Gather first the link fields
    sup_gauge_fields = tf.convert_to_tensor(
        [lt.translate(cfg.gauge_field, mu, +1) for mu in range(cfg.number_of_dimensions)])

    result = 0.
    number_of_plaquettes = 0
    for mu in range(cfg.number_of_dimensions):
        for nu in range(mu + 1, cfg.number_of_dimensions):
            number_of_plaquettes += 1
            result += trace_plaquette(cfg.gauge_field[mu],
                                      sup_gauge_fields[mu, nu],
                                      sup_gauge_fields[nu, mu],
                                      cfg.gauge_field[nu])

    vol = lookup_tables.global_volume

    return Measurement(name="plaquette",
                       value=global_sum(tf.math.real(result)) / (number_of_plaquettes * cfg.colors * vol),
                       geometry=cfg.geometry,
                       colors=cfg.colors)


def temporal_plaquette(cfg: lt.Configuration) -> Measurement:
    '''Return the temporal plaquette expectation value'''
    # Gather first the link fields
    sup_gauge_fields = tf.convert_to_tensor(
        [translate(cfg.gauge_field, mu, +1) for mu in range(cfg.number_of_dimensions)])

    result = 0.
    for mu, nu in [[i, cfg.number_of_dimensions - 1] for i in range(cfg.number_of_dimensions - 1)]:
        result += trace_plaquette(cfg.gauge_field[mu],
                                  sup_gauge_fields[mu, nu],
                                  sup_gauge_fields[nu, mu],
                                  cfg.gauge_field[nu])

    vol = lookup_tables.global_volume

    return Measurement(name="temporal_plaquette",
                       value=global_sum(tf.math.real(result)) / ((cfg.number_of_dimensions - 1) * cfg.colors * vol),
                       geometry=cfg.geometry,
                       colors=cfg.colors)


# ************************************************
# Polyakov and Wilson loops are traces of links along a closed
# path, corresponding to vacuum expectation value of static
# quarks and mesons respectively
# ************************************************

def polyakov_loop_field(cfg, direction):
    '''Return the polyakov loop 3D field along the compactified direction'''
    t_links = cfg.gauge_field[direction]
    polyakov_field = cfg.gauge_field[direction]

    for t in range(cfg.geometry[direction] - 1):
        t_links = translate(t_links, direction, +1)
        polyakov_field = tf.einsum("jkc,kmc->jmc", polyakov_field, t_links)

    return tf.math.add_n([polyakov_field[i, i] for i in range(cfg.colors)])


def wilson_lines_field(cfg, direction, maxT):
    '''Return the Wilson line 3D field along a given direction
    up to length maxT'''
    t_links = cfg.gauge_field[direction]
    lines_field = [cfg.gauge_field[direction]]

    for t in range(maxT):
        t_links = translate(t_links, direction, +1)
        lines_field.append(tf.einsum("jkc,kmc->jmc", lines_field[-1], t_links))

    return lines_field


def wilson_loops(cfg, direction_R, direction_T, max_R, max_T):
    '''Return the expectation value of the Wilson loops along the directions
    direction_R and direction_T up to area max_R * max_T'''
    volume = lookup_tables.global_volume

    R_fields = wilson_lines_field(cfg, direction_R, max_R)
    T_fields = wilson_lines_field(cfg, direction_T, max_T)

    # The Wilson loop is a rectangle
    # So we need to translate the Wilson
    # lines of two farthest sides before computing
    # the trace
    translated_R_fields = [R_fields]
    for t in range(max_T + 1):
        translated_R_fields.append([translate(elem, direction_T, +1) for elem in translated_R_fields[-1]])

    translated_T_fields = [T_fields]
    for r in range(max_R + 1):
        translated_T_fields.append([translate(elem, direction_R, +1) for elem in translated_T_fields[-1]])

    loops = []
    for r in range(1, max_R + 1):
        for t in range(1, max_T + 1):
            loops.append(
                Measurement(
                    name=f"wilson_loop_{r}_{t}",
                    value=global_sum(trace_plaquette(
                        translated_R_fields[0][r - 1],
                        translated_T_fields[r][t - 1],
                        translated_R_fields[t][r - 1],
                        translated_T_fields[0][t - 1],
                    )) / (cfg.colors * volume),
                    geometry=cfg.geometry,
                    colors=cfg.colors
                )
            )
    return loops


def polyakov_operators(cfg: lt.Configuration, correlator_direction=2, polyakov_direction=3):
    '''Return the Polyakov loop, the meson correlator and the thermal polyakov
    loop correlator'''
    coordinates = lookup_tables.global_coordinates

    # First we compute the polyakov loop for each site
    polyakov_field = polyakov_loop_field(cfg, polyakov_direction)

    # The meson field is just the product of two polyakov loops
    # running in opposite directions
    meson_field = polyakov_field * tf.math.conj(translate(polyakov_field, 1, +1))
    meson_zero_momentum_projection = []

    for z in range(cfg.geometry[correlator_direction]):
        # The meson operator is summed for each timeslice separately
        # and only for the origin of the compact dimension
        meson_zero_momentum_projection.append(tf.reduce_mean(tf.gather(meson_field,
                                                                       tf.where(tf.logical_and(
                                                                           coordinates[correlator_direction] == z,
                                                                           coordinates[polyakov_direction] == 0)
                                                                       ))))

    polyakov_correlator = []

    # Correlator measured along the correlator_direction
    polyakov_field_dt = polyakov_field
    for dt in range(cfg.geometry[correlator_direction]):
        # Polyakov loop correlator
        polyakov_correlator.append(tf.reduce_mean(polyakov_field * tf.math.conj(polyakov_field_dt)))
        # Translate the polyakov loop in the z-direction
        polyakov_field_dt = translate(polyakov_field_dt, correlator_direction, +1)

    # Gather the sum from all nodes
    for i in range(cfg.geometry[correlator_direction]):
        polyakov_correlator[i] = global_sum(polyakov_correlator[i])
        meson_zero_momentum_projection[i] = global_sum(meson_zero_momentum_projection[i])

    return [Measurement(name="meson_operator",
                        value=tf.convert_to_tensor(np.array(meson_zero_momentum_projection)),
                        geometry=cfg.geometry,
                        colors=cfg.colors),

            Measurement(name="polyakov_correlator",
                        value=tf.convert_to_tensor(np.array(polyakov_correlator)),
                        geometry=cfg.geometry,
                        colors=cfg.colors),

            Measurement(name="polyakov_loop",
                        value=global_sum(tf.reduce_mean(polyakov_field)),
                        geometry=cfg.geometry,
                        colors=cfg.colors)]


# ************************************************
# The glueball operators are traces of links along closed paths
# and are used to define the correlator for the glueball bound
# state
# ************************************************

@tf.function
def _1x1_plaquette(links1, links2, links3, links4):
    '''Compute the trace of the links in one plaquette'''
    return tf.einsum(
        "klc,lmc,nmc,knc->c",
        links1, links2, tf.math.conj(links3), tf.math.conj(links4))


@tf.function
def _2x2_plaquette(links1, links2, links3, links4, links5, links6, links7, links8):
    return tf.einsum(
        "klc,lmc,mnc,noc,poc,qpc,rqc,krc->c",
        links1, links2, links3, links4,
        tf.math.conj(links5), tf.math.conj(links6),
        tf.math.conj(links7), tf.math.conj(links8))


def glueball_operators(cfg: lt.Configuration, correlator_direction=2):
    '''Compute for each timeslice the 0++ glueball operators,
    required to compute the glueball mass'''
    coordinates = lookup_tables.global_coordinates

    # Gather first the translated link fields
    sup_gauge_fields = tf.convert_to_tensor(
        [translate(cfg.gauge_field, mu, +1) for mu in range(cfg.number_of_dimensions)])

    planes = []
    for mu in range(cfg.number_of_dimensions):
        for nu in range(mu + 1, cfg.number_of_dimensions):
            if mu != correlator_direction and nu != correlator_direction:
                planes.append([mu, nu])

    plaquette_fields = []
    square_fields = []
    for mu, nu in planes:
        # 1x1 plaquette in the mu nu plane
        plaquette_fields.append(_1x1_plaquette(
            cfg.gauge_field[mu],
            sup_gauge_fields[mu, nu],
            sup_gauge_fields[nu, mu],
            cfg.gauge_field[nu]))

        # Further links required to compute the 2x2 plaquette
        sup_mu_mu = translate(sup_gauge_fields[mu, nu], mu, +1)
        sup_mu_mu_nu = translate(sup_mu_mu, nu, +1)
        sup_mu_nu_nu = translate(translate(sup_gauge_fields[mu, mu], nu, +1), nu, +1)
        sup_nu_nu = translate(sup_gauge_fields[nu, mu], nu, +1)

        # 2x2 plaquette in the mu nu plane
        square_fields.append(_2x2_plaquette(
            cfg.gauge_field[mu],
            sup_gauge_fields[mu, mu],
            sup_mu_mu,
            sup_mu_mu_nu,
            sup_mu_nu_nu,
            sup_nu_nu,
            sup_gauge_fields[nu, nu],
            cfg.gauge_field[nu]))

    plaquette_field = tf.math.add_n(plaquette_fields)
    square_field = tf.math.add_n(square_fields)

    plaquette_zero_momentum_projection = []
    square_zero_momentum_projection = []

    for z in range(cfg.geometry[correlator_direction]):
        timeslice = tf.where(coordinates[correlator_direction] == z)
        # If some site of the timeslice lives in the current rank
        if len(timeslice) != 0:
            plaquette_zero_momentum_projection.append(
                tf.reduce_mean(
                    tf.gather(plaquette_field, timeslice)
                )
            )
            square_zero_momentum_projection.append(
                tf.reduce_mean(
                    tf.gather(square_field, timeslice)
                )
            )
        else:
            plaquette_zero_momentum_projection.append(0.)
            square_zero_momentum_projection.append(0.)

    # Gather the sum from all nodes
    for i in range(cfg.geometry[correlator_direction]):
        plaquette_zero_momentum_projection[i] = global_sum(plaquette_zero_momentum_projection[i])
        square_zero_momentum_projection[i] = global_sum(square_zero_momentum_projection[i])

    return [Measurement(name="zpp_operator",
                        value=tf.convert_to_tensor(np.array(plaquette_zero_momentum_projection)),
                        geometry=cfg.geometry,
                        colors=cfg.colors),

            Measurement(name="square_operator",
                        value=tf.convert_to_tensor(np.array(square_zero_momentum_projection)),
                        geometry=cfg.geometry,
                        colors=cfg.colors)]


