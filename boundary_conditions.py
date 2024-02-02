import numpy as np

from translate import lookup_tables
import lattice as lt
import tensorflow as tf

antiperiodic_field_in_t_direction = None


def set_antiperiodic_field_in_t_direction(lattice_geometry: tuple, factor=-1):
    global antiperiodic_field_in_t_direction
    global_coordinates = lookup_tables.global_coordinates

    antiperiodic_field = np.ones_like(global_coordinates[0], dtype=complex)

    for site, coordinate in enumerate(global_coordinates[-1]):
        if coordinate == lattice_geometry[-1] - 1:
            antiperiodic_field[site] = factor

    antiperiodic_field_in_t_direction = tf.convert_to_tensor(antiperiodic_field, dtype=tf.complex128)


def apply_antiboundary_conditions_in_t_direction(cfg):
    if isinstance(cfg, lt.Configuration):
        twisted = cfg.gauge_field[-1]
        twisted = tf.einsum("ijs,s->ijs", twisted, antiperiodic_field_in_t_direction)

        gauge_field = []
        for elem in cfg.gauge_field[:-1]:
            gauge_field.append(elem)
        gauge_field.append(twisted)

        return lt.Configuration(gauge_field=tf.convert_to_tensor(gauge_field, dtype=tf.complex128),
                                geometry=cfg.geometry,
                                colors=cfg.colors,
                                number_of_dimensions=cfg.number_of_dimensions)
    else:
        twisted = cfg[-1]
        twisted = tf.einsum("ijs,s->ijs", twisted, antiperiodic_field_in_t_direction)

        result = []
        for elem in cfg[:-1]:
            result.append(elem)
        result.append(twisted)

        return tf.convert_to_tensor(result, dtype=tf.complex128)
