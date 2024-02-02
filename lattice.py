import tensorflow as tf
from collections import namedtuple
import functools
import numpy as np
import xdrlib

Configuration = namedtuple("Configuration", ["gauge_field", "geometry", "colors", "number_of_dimensions"])


def read_config(cfg_name, lattice_size, Nc=3):
    number_of_dimensions = len(lattice_size)

    with open(cfg_name, "rb") as f:
        p = xdrlib.Unpacker(f.read())

    gauge_config = np.array(
        p.unpack_farray(np.prod(lattice_size) * (Nc - 1) * Nc * number_of_dimensions * 2, p.unpack_float)).reshape(
        (np.prod(lattice_size), number_of_dimensions, Nc, Nc - 1, 2))

    # Assuming the format of Istvan, U* stored in place of U
    loaded_gauge_config = tf.convert_to_tensor(gauge_config[:, :, :, :, 0] - 1j * gauge_config[:, :, :, :, 1])

    gauge_config = []
    for mu in range(number_of_dimensions):
        gauge_config.append([[] for _ in range(Nc)])

        tmp = tf.dtypes.cast(
            1. / tf.sqrt(
                tf.abs(loaded_gauge_config[:, mu, 0, 0]) ** 2
                + tf.abs(loaded_gauge_config[:, mu, 1, 0]) ** 2
                + tf.abs(loaded_gauge_config[:, mu, 2, 0]) ** 2),
            dtype=tf.complex128
        )
        gauge_config[mu][0].append(loaded_gauge_config[:, mu, 0, 0] * tmp)
        gauge_config[mu][1].append(loaded_gauge_config[:, mu, 1, 0] * tmp)
        gauge_config[mu][2].append(loaded_gauge_config[:, mu, 2, 0] * tmp)

        tmp = (
                loaded_gauge_config[:, mu, 0, 1] * tf.math.conj(gauge_config[mu][0][0])
                + loaded_gauge_config[:, mu, 1, 1] * tf.math.conj(gauge_config[mu][1][0])
                + loaded_gauge_config[:, mu, 2, 1] * tf.math.conj(gauge_config[mu][2][0]))
        gauge_config[mu][0].append(loaded_gauge_config[:, mu, 0, 1] - tmp * gauge_config[mu][0][0])
        gauge_config[mu][1].append(loaded_gauge_config[:, mu, 1, 1] - tmp * gauge_config[mu][1][0])
        gauge_config[mu][2].append(loaded_gauge_config[:, mu, 2, 1] - tmp * gauge_config[mu][2][0])

        tmp = tf.dtypes.cast(
            1 / tf.sqrt(
                tf.abs(gauge_config[mu][0][1]) ** 2
                + tf.abs(gauge_config[mu][1][1]) ** 2
                + tf.abs(gauge_config[mu][2][1]) ** 2),
            dtype=tf.complex128
        )
        gauge_config[mu][0][1] = gauge_config[mu][0][1] * tmp
        gauge_config[mu][1][1] = gauge_config[mu][1][1] * tmp
        gauge_config[mu][2][1] = gauge_config[mu][2][1] * tmp

        gauge_config[mu][0].append(
            tf.math.conj(gauge_config[mu][1][0]) * tf.math.conj(gauge_config[mu][2][1]) - tf.math.conj(
                gauge_config[mu][2][0]) * tf.math.conj(gauge_config[mu][1][1]))
        gauge_config[mu][1].append(
            tf.math.conj(gauge_config[mu][2][0]) * tf.math.conj(gauge_config[mu][0][1]) - tf.math.conj(
                gauge_config[mu][0][0]) * tf.math.conj(gauge_config[mu][2][1]))
        gauge_config[mu][2].append(
            tf.math.conj(gauge_config[mu][0][0]) * tf.math.conj(gauge_config[mu][1][1]) - tf.math.conj(
                gauge_config[mu][1][0]) * tf.math.conj(gauge_config[mu][0][1]))

    print(cfg_name, [p.unpack_int() for _ in range(number_of_dimensions)], ", params",
          [p.unpack_double() for _ in range(3)])
    return Configuration(gauge_field=tf.convert_to_tensor(gauge_config, dtype=tf.complex128), geometry=lattice_size,
                         colors=Nc)


# ************************************************
# In multithreading mode we need to know for each site
# its corresponding neighbors and its cartesian global
# coordinate.
# ************************************************

MthLookupTables = namedtuple("MthLookupTables", ["sup",
                                                 "sdn",
                                                 "global_coordinates",
                                                 "global_volume",
                                                 "local_volume"])


@functools.lru_cache(maxsize=100)
def mth_lookup_tables(lattice_geometry: tuple):
    """General function to compute look-up tables for the nearest neighbors and the global coordinates of a site"""
    # Global coordinates of a given global lattice site
    number_of_dimensions = len(lattice_geometry)

    global_coordinates = np.empty(lattice_geometry + (number_of_dimensions,), dtype=np.int64)

    coordinate_iterator = (np.arange(0, L) for L in lattice_geometry)
    for i, a in enumerate(np.ix_(*coordinate_iterator)):
        global_coordinates[..., i] = a

    global_coordinates = global_coordinates.reshape(-1, number_of_dimensions).T

    # Sup and down nearest neighbor for each site
    sup = np.zeros(shape=(number_of_dimensions, np.prod(lattice_geometry)), dtype=np.int64)
    sdn = np.zeros(shape=(number_of_dimensions, np.prod(lattice_geometry)), dtype=np.int64)
    for mu in range(number_of_dimensions):
        sup_coordinates = global_coordinates.copy()
        sup_coordinates[mu] = (sup_coordinates[mu] + 1) % lattice_geometry[mu]

        for nu in range(number_of_dimensions):
            sup[mu] = lattice_geometry[nu] * sup[mu] + sup_coordinates[nu] % lattice_geometry[nu]

        sdn_coordinates = global_coordinates.copy()
        sdn_coordinates[mu] = (sdn_coordinates[mu] - 1) % lattice_geometry[mu]

        for nu in range(number_of_dimensions):
            sdn[mu] = lattice_geometry[nu] * sdn[mu] + sdn_coordinates[nu] % lattice_geometry[nu]

    return MthLookupTables(tf.convert_to_tensor(sup),
                           tf.convert_to_tensor(sdn),
                           tf.convert_to_tensor(global_coordinates),
                           np.prod(lattice_geometry),
                           np.prod(lattice_geometry))


#@tf.function
def mth_translate(tensor, direction, sign, lookup_tables):
    if sign > 0:
        return tf.gather(tensor, lookup_tables.sup[direction], axis=len(tensor.shape) - 1)
    else:
        return tf.gather(tensor, lookup_tables.sdn[direction], axis=len(tensor.shape) - 1)


# ************************************************
# In MPI mode we need also to store the nearest neighbors
# inside each MPI rank and between the MPI ranks through 
# the boundaries
# ************************************************

MPILookupTables = namedtuple("MPILookupTables", ["sup",
                                                 "sdn",
                                                 "global_coordinates",
                                                 "to_send_sup",
                                                 "to_receive_sup",
                                                 "to_send_sdn",
                                                 "to_receive_sdn",
                                                 "global_volume",
                                                 "local_volume"
                                                 ])


@functools.lru_cache(maxsize=100)
def MPI_lookup_tables(lattice_geometry: tuple, pgrid_size: tuple) -> MPILookupTables:
    '''General function to compute look-up tables for the nearest neighbors
    and the global coordinates of a site in MPI mode'''

    from mpi4py import MPI

    number_of_dimensions = len(lattice_geometry)

    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()

    for d in number_of_dimensions:
        assert lattice_geometry[d] % pgrid_size[d] == 0, f"L{d} not divisible by pgrid_{d}"

    loc = [L / g for L, g in zip(lattice_geometry, pgrid_size)]

    local_volume = np.prod(pgrid_size)
    global_volume = np.prod(lattice_geometry)

    # Global coordinates of a given global lattice site
    global_coordinates = np.empty(lattice_geometry + (number_of_dimensions,), dtype=np.int64)
    coordinate_iterator = (np.arange(0, L) for L in lattice_geometry)
    for i, a in enumerate(np.ix_(*coordinate_iterator)):
        global_coordinates[..., i] = a

    global_coordinates = global_coordinates.reshape(-1, number_of_dimensions).T

    # Rank table containing, for each global site, the corresponding rank
    rank_table = global_coordinates[0] // loc[0]
    for mu in range(1, number_of_dimensions):
        rank_table = pgrid_size[mu] * rank_table + global_coordinates[mu] // loc[mu]

    # Global coordinates of a given local lattice site
    local_coordinates = np.zeros(shape=(number_of_dimensions, local_volume), dtype=np.int64)
    for mu in range(0, number_of_dimensions):
        local_coordinates[mu] = global_coordinates[mu, np.where(rank_table == my_rank)]

    # Sup and down nearest neighbor for the sites
    # inside each MPI block lattice
    # The sup and down site that are going out the
    # local sublattice are fixed by MPI communications
    sup = np.zeros(shape=(number_of_dimensions, local_volume), dtype=np.int64)
    sdn = np.zeros(shape=(number_of_dimensions, local_volume), dtype=np.int64)
    for mu in range(number_of_dimensions):
        sup_coordinates = global_coordinates.copy()
        sup_coordinates[mu] = (sup_coordinates[mu] + 1) % lattice_geometry[mu]

        sup_tmp = np.zeros(shape=np.prod(lattice_geometry), dtype=np.int64)
        for nu in range(number_of_dimensions):
            sup_tmp = lattice_geometry[nu] * sup_tmp + sup_coordinates[nu] % lattice_geometry[nu]

        sup[mu] = sup_tmp[np.where(rank_table == my_rank)]

        sdn_coordinates = global_coordinates.copy()
        sdn_coordinates[mu] = (sdn_coordinates[mu] - 1) % lattice_geometry[mu]

        sdn_tmp = np.zeros(shape=np.prod(lattice_geometry), dtype=np.int64)
        for nu in range(number_of_dimensions):
            sdn_tmp = lattice_geometry[nu] * sdn_tmp + sdn_coordinates[nu] % lattice_geometry[nu]

        sdn[mu] = sdn_tmp[np.where(rank_table == my_rank)]

    number_of_processors = np.prod(pgrid_size)

    # Sites to be sent and received by each node
    to_send_sup = [[[] for _ in range(number_of_processors)] for _ in range(number_of_dimensions)]
    to_send_sdn = [[[] for _ in range(number_of_processors)] for _ in range(number_of_dimensions)]
    to_receive_sup = [[[] for _ in range(number_of_processors)] for _ in range(number_of_dimensions)]
    to_receive_sdn = [[[] for _ in range(number_of_processors)] for _ in range(number_of_dimensions)]

    for mu in range(number_of_dimensions):
        sup_coordinates = global_coordinates.copy()
        sup_coordinates[mu] = (sup_coordinates[mu] + 1) % lattice_geometry[mu]

        sup_rank_table = sup_coordinates[0] // loc[0]

        for nu in range(1, number_of_dimensions):
            sup_rank_table = pgrid_size[nu] * sup_rank_table + sup_coordinates[nu] // loc[nu]

        # If the sup site lives in an another MPI block
        # If I need to send my site
        for site in np.where(np.logical_and(sup_rank_table != rank_table, sup_rank_table == my_rank))[0]:
            this_site_rank = rank_table[site]
            site_coordinate = sup_coordinates[:, site]
            site_index = 0
            for nu in range(number_of_dimensions):
                site_index = loc[nu] * site_index + site_coordinate[nu] % loc[nu]

            to_send_sup[mu][this_site_rank].append(site_index)

        # If a will receive the site from somebody else
        for site in np.where(np.logical_and(sup_rank_table != rank_table, rank_table == my_rank))[0]:
            sup_rank = sup_rank_table[site]
            site_coordinate = sup_coordinates[:, site]
            site_index = 0
            for nu in range(number_of_dimensions):
                site_index = loc[nu] * site_index + site_coordinate[nu] % loc[nu]

            to_receive_sup[mu][sup_rank].append(site_index)

        sdn_coordinates = global_coordinates.copy()
        sdn_coordinates[mu] = (sdn_coordinates[mu] - 1) % lattice_geometry[mu]

        sdn_rank_table = sdn_coordinates[0] // loc[0]
        for nu in range(1, number_of_dimensions):
            sdn_rank_table = pgrid_size[nu] * sdn_rank_table + sdn_coordinates[nu] // loc[nu]

        # If the sdn site lives in an another MPI block
        # If I need to send my site
        for site in np.where(np.logical_and(sdn_rank_table != rank_table, sdn_rank_table == my_rank))[0]:
            this_site_rank = rank_table[site]
            site_coordinate = sdn_coordinates[:, site]
            site_index = 0
            for nu in range(number_of_dimensions):
                site_index = loc[nu] * site_index + site_coordinate[nu] % loc[nu]

            to_send_sdn[mu][this_site_rank].append(site_index)

        # If a will receive the site from somebody else
        for site in np.where(np.logical_and(sdn_rank_table != rank_table, rank_table == my_rank))[0]:
            sdn_rank = sdn_rank_table[site]
            site_coordinate = sdn_coordinates[:, site]
            site_index = 0
            for nu in range(number_of_dimensions):
                site_index = loc[nu] * site_index + site_coordinate[nu] % loc[nu]

            to_receive_sdn[mu][sdn_rank].append(site_index)

    return MPILookupTables(tf.convert_to_tensor(sup),
                           tf.convert_to_tensor(sdn),
                           tf.convert_to_tensor(local_coordinates),
                           np.array(to_send_sup),
                           np.array(to_receive_sup),
                           np.array(to_send_sdn),
                           np.array(to_receive_sdn),
                           global_volume,
                           local_volume)


# ************************************************
# In MPI mode the translation of a lattice in one direction
# requires a reshuflling of the local sites and a lot of MPI
# communications beween the MPI ranks
# ************************************************

def MPI_translate(tensor, direction, sign, lattice_geometry, pgrid_size):
    '''Translate a given tensor along direction forward (positive) 
    or backward (negative sign) of one lattice site in MPI mode'''
    lt = MPI_lookup_tables(lattice_geometry, pgrid_size)

    if sign > 0:
        # If the shift does not require MPI communications
        if np.sum(lt.to_send_sup[direction]) == 0:
            return tf.gather(tensor, lt.sup[direction], axis=len(tensor.shape) - 1)
        else:
            # Otherwise
            comm = MPI.COMM_WORLD

            # First translate the site that are inside each block
            local_exchange = tf.transpose(
                tf.gather(tensor, lt.sup[direction], axis=len(tensor.shape) - 1),
                np.roll(range(len(tensor.shape)), 1)).numpy()

            # Original tensor to be exchanged
            to_send = tf.transpose(tensor, np.roll(range(len(tensor.shape)), 1)).numpy()

            req_send = []
            req_recv = []

            for rank, indexes in enumerate(lt.to_send_sup[direction]):
                # First we send our neightbors
                if len(indexes) != 0:
                    req_send.append(comm.isend(to_send[indexes], dest=rank, tag=11))
            for rank, indexes in enumerate(lt.to_receive_sup[direction]):
                # then we receive our neightbors
                if len(indexes) != 0:
                    buf = bytearray(128 * np.prod(tensor.shape))
                    req_recv.append([comm.irecv(buf, source=rank, tag=11), indexes])

            # Wait for the communications
            for req in req_send:
                req.wait()
            for req, indexes in req_recv:
                local_exchange[indexes] = req.wait()

            # Translate indeces and return
            return tf.transpose(tf.convert_to_tensor(local_exchange), np.roll(range(len(tensor.shape)), -1))
    else:
        # do the same for the down indeces
        if np.sum(lt.to_send_sdn[direction]) == 0:
            return tf.gather(tensor, lt.sdn[direction], axis=len(tensor.shape) - 1)
        else:
            comm = MPI.COMM_WORLD

            local_exchange = tf.transpose(
                tf.gather(tensor, lt.sdn[direction], axis=len(tensor.shape) - 1),
                np.roll(range(len(tensor.shape)), 1)).numpy()

            to_send = tf.transpose(tensor, np.roll(range(len(tensor.shape)), 1)).numpy()

            req_send = []
            req_recv = []

            for rank, indexes in enumerate(lt.to_send_sdn[direction]):
                if len(indexes) != 0:
                    req_send.append(comm.isend(to_send[indexes], dest=rank, tag=11))
            for rank, indexes in enumerate(lt.to_receive_sdn[direction]):
                if len(indexes) != 0:
                    buf = bytearray(128 * np.prod(tensor.shape))
                    req_recv.append([comm.irecv(buf, source=rank, tag=11), indexes])

            for req in req_send:
                req.wait()
            for req, indexes in req_recv:
                local_exchange[indexes] = req.wait()

            return tf.transpose(tf.convert_to_tensor(local_exchange), np.roll(range(len(tensor.shape)), -1))


# ************************************************
# In MPI mode we need to collect the results from all ranks
# when summing an observable over the entire lattice.
# We therefore implement utility functions for it
# ************************************************

def mth_global_sum(to_sum):
    return to_sum


def MPI_global_sum(to_sum):
    '''Perform the Allreduce of to_sum for all ranks'''
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()

    # First convert to numpy if needed
    try:
        to_sum = to_sum.numpy()
    except AttributeError:
        pass

    # Setup the buffers for the Allreduce
    local_buffer = np.zeros(shape=2)
    global_buffer = np.zeros(shape=2)

    local_buffer[0] = to_sum.real
    local_buffer[1] = to_sum.imag

    comm.Allreduce(local_buffer, global_buffer, MPI.SUM)

    # return a real or complex number
    if np.iscomplex(global_buffer[0] + 1j * global_buffer[1]):
        return global_buffer[0] + 1j * global_buffer[1]
    else:
        return global_buffer[0]


# ************************************************
# In MPI mode we need to test that the communications
# between the ranks are correct. We perform some shift
# and check that the lattice is translated correctly.
# ************************************************

def test_MPI_communications(lattice_geometry, pgrid_size):
    lt = MPI_lookup_tables(lattice_geometry, pgrid_size)

    number_of_dimensions = len(lattice_geometry)

    for mu in range(number_of_dimensions):
        # Some test tensor
        test = np.zeros(shape=(2, 2, lt.local_volume))
        for site in range(lt.local_volume):
            test[0, 0, site] = lt.global_coordinates[mu][site]

        prev_sum = MPI_global_sum(np.sum(test))

        test = tf.convert_to_tensor(test)

        translated_sup = MPI_translate(test, mu, +1, lattice_geometry, pgrid_size).numpy()
        translated_sdn = MPI_translate(test, mu, -1, lattice_geometry, pgrid_size).numpy()

        sup_sum = MPI_global_sum(np.sum(translated_sup))
        sdn_sum = MPI_global_sum(np.sum(translated_sdn))

        if sup_sum != prev_sum or sdn_sum != prev_sum:
            print("Sum of the translated lattices differs!")
            return False

        for site in range(lt.local_volume):
            if translated_sup[0, 0, site] != (lt.global_coordinates[mu][site] + 1) % lattice_geometry[mu]:
                print("Communication error in rank", my_rank, "mu", mu, "sign up")
                return False
            if translated_sdn[0, 0, site] != (lt.global_coordinates[mu][site] - 1) % lattice_geometry[mu]:
                print("Communication error in rank", my_rank, "mu", mu, "sign down")
                return False

    print("MPI Communication test passed")
    return True
