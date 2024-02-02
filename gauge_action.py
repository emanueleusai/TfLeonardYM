import tensorflow as tf
import lattice as lt


class yang_mills:
    def __init__(self, cfg: lt.Configuration, beta):
        self.beta = beta
        self.cfg = cfg

    def set_gauge_configuration(self, cfg):
        self.cfg = cfg

    def force(self):
        """Compute the gauge force for a gauge coupling beta"""

        # ************************************************
        # The gauge force, required for to integrate the equation of 
        # motion, is field equal to the sum of the staples.
        # ************************************************
        staples = tf.convert_to_tensor([
            tf.math.accumulate_n(elem)
            for elem in _get_staple_fields(self.cfg, no_direction=self.cfg.number_of_dimensions)])

        return _gauge_force_from_staples(self.cfg, staples, self.beta)

    def energy(self):
        # Gather first the link fields
        sup_gauge_fields = tf.convert_to_tensor(
            [lt.translate(self.cfg.gauge_field, mu, +1) for mu in range(self.cfg.number_of_dimensions)])

        gauge_energy = 0.
        for mu in range(self.cfg.number_of_dimensions):
            for nu in range(mu + 1, self.cfg.number_of_dimensions):
                gauge_energy += trace_plaquette(self.cfg.gauge_field[mu],
                                                sup_gauge_fields[mu, nu],
                                                sup_gauge_fields[nu, mu],
                                                self.cfg.gauge_field[nu])

        return -self.beta * tf.math.real(gauge_energy) / self.cfg.colors


@tf.function
def _gauge_force_from_staples(cfg: lt.Configuration, staples, beta):
    '''Compute the gauge force given the precomputed staples field'''
    plaquette = tf.einsum("mijc,mkjc->mikc", cfg.gauge_field, tf.math.conj(staples))
    force = (-0.25 * beta / cfg.colors) * (
            tf.math.conj(tf.transpose(plaquette, [0, 2, 1, 3])) - plaquette)
    trace = tf.add_n([force[:, i, i] for i in range(cfg.colors)]) / cfg.colors
    return tf.convert_to_tensor([[
        [force[mu, i, j] if i != j else force[mu, i, i] - trace[mu]
         for j in range(cfg.colors)]
        for i in range(cfg.colors)]
        for mu in range(cfg.number_of_dimensions)])


def _get_staple_fields(cfg: lt.Configuration, no_direction: int = 3):
    """Compute the sum of the links in each plane for the staples around
    each direction"""
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
            translated_field[(mu, nu, +1)] = lt.translate(cfg.gauge_field[mu], nu, +1)
        if not (nu, mu, +1) in translated_field:
            translated_field[(nu, mu, +1)] = lt.translate(cfg.gauge_field[nu], mu, +1)
        if not (mu, nu, -1) in translated_field:
            translated_field[(mu, nu, -1)] = lt.translate(cfg.gauge_field[mu], nu, -1)
        if not (nu, nu, -1) in translated_field:
            translated_field[(nu, nu, -1)] = lt.translate(cfg.gauge_field[nu], nu, -1)
        if not (nu, nu, -1, mu, +1) in translated_field:
            translated_field[(nu, nu, -1, mu, +1)] = lt.translate(translated_field[(nu, nu, -1)], mu, +1)

    staple_fields = [[] for _ in range(cfg.number_of_dimensions)]
    for mu, nu in planes:
        staple_fields[mu].append(_sup_staple_trace(cfg.gauge_field[nu],
                                                   translated_field[(mu, nu, +1)],
                                                   translated_field[(nu, mu, +1)]))

        staple_fields[mu].append(_sdn_staple_trace(translated_field[(nu, nu, -1)],
                                                   translated_field[(mu, nu, -1)],
                                                   translated_field[(nu, nu, -1, mu, +1)]))

    return staple_fields


@tf.function
def _sup_staple_trace(links1, links2, links3):
    return tf.einsum("kmc,mnc,pnc->kpc", links1, links2, tf.math.conj(links3))


@tf.function
def _sdn_staple_trace(links1, links2, links3):
    return tf.einsum("mkc,mnc,npc->kpc", tf.math.conj(links1), links2, links3)


# ************************************************
# The plaquette corresponds to the trace of the square of the 
# field strength tensor F_\mu\nu^ab , and it is the basic
# ingredient to define the gauge action
# ************************************************

@tf.function
def trace_plaquette(links1, links2, links3, links4):
    '''Compute the trace of the links in one plaquette'''
    return tf.einsum("klc,lmc,nmc,knc", links1, links2, tf.math.conj(links3), tf.math.conj(links4))
