try:
    from mpi4py import MPI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    class _DummyComm:
        def Get_rank(self):
            return 0

        def Get_size(self):
            return 1

        def Allreduce(self, sendbuf, recvbuf, op=None):
            import numpy as _np
            _np.copyto(recvbuf, sendbuf)

        def Bcast(self, x, root=0):
            pass

    class _DummyMPI:
        COMM_WORLD = _DummyComm()
        SUM = None
        MIN = None
        MAX = None

    MPI = _DummyMPI()
import os, subprocess, sys
import numpy as np


def mpi_fork(n, bind_to_core=False):
    """
    Re-launches the current script with workers linked by MPI.

    Also, terminates the original process that launched it.

    Taken almost without modification from the Baselines function of the
    `same name`_.

    .. _`same name`: https://github.com/openai/baselines/blob/master/baselines/common/mpi_fork.py

    Args:
        n (int): Number of process to split into.

        bind_to_core (bool): Bind each MPI process to a core.
    """
    if n<=1: 
        return
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        args = ["mpirun", "-np", str(n)]
        if bind_to_core:
            args += ["-bind-to", "core"]
        args += [sys.executable] + sys.argv
        subprocess.check_call(args, env=env)
        sys.exit()


def msg(m, string=''):
    rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
    print(('Message from %d: %s \t ' % (rank, string)) + str(m))

def proc_id():
    """Get rank of calling process."""
    return MPI.COMM_WORLD.Get_rank() if MPI else 0

def allreduce(*args, **kwargs):
    if MPI:
        return MPI.COMM_WORLD.Allreduce(*args, **kwargs)
    if len(args) >= 2:
        import numpy as _np
        _np.copyto(args[1], args[0])
    return args[1] if len(args) >= 2 else None

def num_procs():
    """Count active MPI processes."""
    return MPI.COMM_WORLD.Get_size() if MPI else 1

def broadcast(x, root=0):
    if MPI:
        MPI.COMM_WORLD.Bcast(x, root=root)

def mpi_op(x, op):
    x, scalar = ([x], True) if np.isscalar(x) else (x, False)
    x = np.asarray(x, dtype=np.float32)
    buff = np.zeros_like(x, dtype=np.float32)
    if MPI:
        allreduce(x, buff, op=op)
    else:
        np.copyto(buff, x)
    return buff[0] if scalar else buff

def mpi_sum(x):
    return mpi_op(x, MPI.SUM)

def mpi_avg(x):
    """Average a scalar or vector over MPI processes."""
    return mpi_sum(x) / num_procs()
    
def mpi_statistics_scalar(x, with_min_and_max=False):
    """
    Get mean/std and optional min/max of scalar x across MPI processes.

    Args:
        x: An array containing samples of the scalar to produce statistics
            for.

        with_min_and_max (bool): If true, return min and max of x in 
            addition to mean and std.
    """
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = mpi_sum([np.sum(x), len(x)])
    mean = global_sum / global_n

    global_sum_sq = mpi_sum(np.sum((x - mean)**2))
    std = np.sqrt(global_sum_sq / global_n)  # compute global std

    if with_min_and_max:
        global_min = mpi_op(np.min(x) if len(x) > 0 else np.inf, op=MPI.MIN if MPI else None)
        global_max = mpi_op(np.max(x) if len(x) > 0 else -np.inf, op=MPI.MAX if MPI else None)
        return mean, std, global_min, global_max
    return mean, std
