import tensorflow as tf
from translate import translate
import lattice as lt


# ************************************************
# "Staples" are the sum of the product of links of the
# plaquettes attached around a given link.
# It is required to smear a given configuration and to compute
# the gauge force for the Hybrid Monte Carlo.
# ************************************************

@tf.function
def sup_staple_trace(links1, links2, links3):
    return tf.einsum("kmc,mnc,pnc->kpc", links1, links2, tf.math.conj(links3))


@tf.function
def sdn_staple_trace(links1, links2, links3):
    return tf.einsum("mkc,mnc,npc->kpc", tf.math.conj(links1), links2, links3)


def get_staple_fields(cfg: lt.Configuration, no_direction: int = 3):
    '''Compute the sum of the links in each plane for the staples around
    each direction'''
    # The planes where we compute the staple
    planes = []
    for mu in range(cfg.number_of_dimensions):
        for nu in range(cfg.number_of_dimensions):
            if mu != no_direction and nu != no_direction and mu != nu:
                planes.append([mu, nu])

    translated_field = {}

    # First collect the required translated fields
    for mu, nu in planes:
        if not (mu, nu, +1) in translated_field:
            translated_field[(mu, nu, +1)] = translate(cfg.gauge_field[mu], nu, +1)
        if not (nu, mu, +1) in translated_field:
            translated_field[(nu, mu, +1)] = translate(cfg.gauge_field[nu], mu, +1)
        if not (mu, nu, -1) in translated_field:
            translated_field[(mu, nu, -1)] = translate(cfg.gauge_field[mu], nu, -1)
        if not (nu, nu, -1) in translated_field:
            translated_field[(nu, nu, -1)] = translate(cfg.gauge_field[nu], nu, -1)
        if not (nu, nu, -1, mu, +1) in translated_field:
            translated_field[(nu, nu, -1, mu, +1)] = translate(translated_field[(nu, nu, -1)], mu, +1)

    staple_fields = [[] for _ in range(cfg.number_of_dimensions)]
    for mu, nu in planes:
        staple_fields[mu].append(sup_staple_trace(cfg.gauge_field[nu],
                                                  translated_field[(mu, nu, +1)],
                                                  translated_field[(nu, mu, +1)]))

        staple_fields[mu].append(sdn_staple_trace(translated_field[(nu, nu, -1)],
                                                  translated_field[(mu, nu, -1)],
                                                  translated_field[(nu, nu, -1, mu, +1)]))

    return staple_fields
