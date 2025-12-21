"""Utility functions for parallel computing."""

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False


def is_master():
    """Check if this is the master process (rank 0)."""
    if not MPI_AVAILABLE:
        return True
    return MPI.COMM_WORLD.Get_rank() == 0


def get_rank():
    """Get the current MPI rank."""
    if not MPI_AVAILABLE:
        return 0
    return MPI.COMM_WORLD.Get_rank()


def get_size():
    """Get the total number of MPI processes."""
    if not MPI_AVAILABLE:
        return 1
    return MPI.COMM_WORLD.Get_size()


def print_master(*args, **kwargs):
    """Print only from master process."""
    if is_master():
        print(*args, **kwargs)
