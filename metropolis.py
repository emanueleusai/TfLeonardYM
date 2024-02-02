import numpy as np

metropolis = None


# ************************************************
# Metropolis accept/reject step. In MPI mode, the metropolis
# step is done by the master rank
# ************************************************

def mth_metropolis(new_energy, old_energy):
    '''Metropolis step in single task mode'''
    if new_energy < old_energy:
        return True
    elif np.random.uniform(0, 1) < np.exp(-(new_energy - old_energy)):
        return True
    else:
        return False


def MPI_metropolis(new_energy, old_energy):
    from mpi4py import MPI

    '''Metropolis step in MPI mode'''
    if new_energy < old_energy:
        return True
    else:
        comm = MPI.COMM_WORLD
        my_rank = comm.Get_rank()

        random_uniform = None
        if my_rank == 0:
            random_uniform = np.random.uniform(0, 1)

        random_uniform = comm.bcast(random_uniform, root=0)

        if random_uniform < np.exp(-(new_energy - old_energy)):
            return True
        else:
            return False


def set_metropolis(use_mpi):
    global metropolis
    if use_mpi:
        metropolis = MPI_metropolis
    else:
        metropolis = mth_metropolis
