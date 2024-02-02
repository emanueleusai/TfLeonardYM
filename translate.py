import lattice as lt
import functools

translate = None
lookup_tables = None
global_sum = None


def set_translate(use_mpi: True, geometry, mpi_grid):
    global translate
    global lookup_tables
    global global_sum
    if use_mpi:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        my_rank = comm.Get_rank()

        lt.test_MPI_communications(geometry, mpi_grid)

        lt.translate = functools.partial(lt.MPI_translate, lattice_size=geometry, pgrid_size=mpi_grid)
        translate = lt.translate
        lookup_tables = lt.MPI_lookup_tables(geometry, mpi_grid)
        global_sum = lt.MPI_global_sum
        lt.global_sum = lt.MPI_global_sum
        print = MPI_output

        print("Lattice size:", *geometry)
        print("MPI grid:", *mpi_grid)
    else:
        lt.global_sum = lt.mth_global_sum
        global_sum = lt.mth_global_sum

        lookup_tables = lt.mth_lookup_tables(tuple(geometry))
        lt.translate = functools.partial(lt.mth_translate, lookup_tables=lookup_tables)
        translate = lt.translate

