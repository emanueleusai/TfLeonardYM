import tensorflow as tf
from measurements import Measurement
from staples import get_staple_fields
import lattice as lt



# ************************************************
# Smearing is required to suppress ultraviolet fluctuations to
# improve the signal of operators. It is basically a Gaussian
# filtering applied to the gluon field. The smearing is applied in
# a gauge invariant way using staples, and reunitarizing the
# links at the end
# ************************************************


@tf.function
def unitary_sqrt_projection(link_config):
    """Reunit an SU(N) tensor field"""
    rows = link_config.shape[1]
    number_of_dimensions = link_config.shape[0]
    XXdag = tf.einsum("jmkc,jmnc->jknc", tf.math.conj(link_config), link_config)
    swapped = tf.linalg.inv(tf.linalg.sqrtm(tf.transpose(XXdag, [3, 0, 1, 2])))
    unitary = tf.einsum("jmkc,cjkn->cjmn", link_config, swapped)
    det = tf.linalg.det(unitary)
    unitary = tf.convert_to_tensor([[[
        unitary[:, mu, j, i] / (det[:, mu] ** (1 / rows))
        for i in range(rows)]
        for j in range(rows)]
        for mu in range(number_of_dimensions)])
    return unitary


def ape_smearing(cfg, rho, no_smear_direction=3):
    '''Smear a gauge field cfg using a smearing parameter rho.
    For correlation functions, links must not be smeared in the
    time direction.'''
    staple_fields = get_staple_fields(cfg, no_smear_direction)

    fat_links = []
    for mu, elem in enumerate(staple_fields):
        if len(elem) != 0:
            fat_links.append((rho / 6) * tf.math.add_n(elem))
        else:
            fat_links.append(rho * cfg.gauge_field[mu])

    unitary_links = unitary_sqrt_projection(tf.convert_to_tensor(fat_links) + (1 - rho) * cfg.gauge_field)

    return lt.Configuration(gauge_field=unitary_links,
                            geometry=cfg.geometry,
                            colors=cfg.colors)


def multi_smearing_operators(cfg: lt.Configuration,
                             operators_to_measure,
                             smearing_levels=10,
                             rho=0.3,
                             no_smear_direction=3,
                             skip=1):
    '''Smear a gauge field cfg using a smearing parameter rho
    and measure consecutively the operators in operators_to_measure
    for n smearing_levels after each skip steps.
    For correlation functions, links must not be smeared in the
    time direction.'''
    result = []
    next_cfg = cfg

    # First measure on the unsmeared links
    for operator in operators_to_measure:
        result.extend(operator(next_cfg))

    for level in range(smearing_levels // skip):
        # Now we proceed to smear the links
        for step in range(skip):
            next_cfg = ape_smearing(next_cfg, rho, no_smear_direction=no_smear_direction)
        # ... and we measure on the smeared links
        for operator in operators_to_measure:
            measurements = operator(next_cfg)
            for meas in measurements:
                result.append(Measurement(name=meas.name,
                                          value=meas.value,
                                          geometry=meas.geometry,
                                          colors=meas.colors,
                                          smearing_levels=level))

    return result

