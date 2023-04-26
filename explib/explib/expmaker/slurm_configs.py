from functools import partial


def set_config(time, gpu, mem, cpus):
    return {
        "gpu": gpu,
        "mem": mem,
        "time": time,
        "cpus-per-task": cpus,
    }


small_cpu = partial(set_config, gpu=None, mem="12000M", cpus=2)
medium_cpu = partial(set_config, gpu=None, mem="32000M", cpus=8)
small_gpu = partial(set_config, gpu="p100", mem="16000M", cpus=6)
large_gpu = partial(set_config, gpu="v100l", mem="48000M", cpus=8)
default_gpu = partial(set_config, gpu="1", mem="30000M", cpus=5)
narval_gpu = partial(set_config, gpu="a100", mem="48000M", cpus=8)

SMALL_CPU_2H = "SMALL_CPU_2H"
SMALL_CPU_4H = "SMALL_CPU_4H"
SMALL_CPU_8H = "SMALL_CPU_8H"
SMALL_CPU_16H = "SMALL_CPU_16H"
SMALL_CPU_24H = "SMALL_CPU_24H"
MEDIUM_CPU_2H = "MEDIUM_CPU_2H"
MEDIUM_CPU_4H = "MEDIUM_CPU_4H"
MEDIUM_CPU_8H = "MEDIUM_CPU_8H"
MEDIUM_CPU_16H = "MEDIUM_CPU_16H"
MEDIUM_CPU_24H = "MEDIUM_CPU_24H"
DEFAULT_GPU_2H = "DEFAULT_GPU_2H"
DEFAULT_GPU_4H = "DEFAULT_GPU_4H"
DEFAULT_GPU_8H = "DEFAULT_GPU_8H"
DEFAULT_GPU_12H = "DEFAULT_GPU_12H"
DEFAULT_GPU_16H = "DEFAULT_GPU_16H"
DEFAULT_GPU_24H = "DEFAULT_GPU_24H"
DEFAULT_GPU_36H = "DEFAULT_GPU_36H"
NARVAL_GPU_2H = "NARVAL_GPU_2H"
NARVAL_GPU_4H = "NARVAL_GPU_4H"
NARVAL_GPU_8H = "NARVAL_GPU_8H"
NARVAL_GPU_16H = "NARVAL_GPU_16H"
NARVAL_GPU_24H = "NARVAL_GPU_24H"
SMALL_GPU_1H = "SMALL_GPU_1H"
SMALL_GPU_2H = "SMALL_GPU_2H"
SMALL_GPU_4H = "SMALL_GPU_4H"
SMALL_GPU_8H = "SMALL_GPU_8H"
SMALL_GPU_12H = "SMALL_GPU_12H"
SMALL_GPU_16H = "SMALL_GPU_16H"
LARGE_GPU_1H = "LARGE_GPU_1H"
LARGE_GPU_2H = "LARGE_GPU_2H"
LARGE_GPU_2HALFH = "LARGE_GPU_2HALFH"
LARGE_GPU_4H = "LARGE_GPU_4H"
LARGE_GPU_8H = "LARGE_GPU_8H"
LARGE_GPU_6H = "LARGE_GPU_6H"
LARGE_GPU_12H = "LARGE_GPU_12H"
LARGE_GPU_16H = "LARGE_GPU_16H"
LARGE_GPU_24H = "LARGE_GPU_24H"
LARGE_GPU_36H = "LARGE_GPU_36H"
LARGE_GPU_72H = "LARGE_GPU_72H"

SLURM_CONFIGS = {
    SMALL_CPU_2H: small_cpu("0-02:00"),
    SMALL_CPU_4H: small_cpu("0-04:00"),
    SMALL_CPU_8H: small_cpu("0-08:00"),
    SMALL_CPU_16H: small_cpu("0-16:00"),
    SMALL_CPU_24H: small_cpu("0-24:00"),
    MEDIUM_CPU_2H: medium_cpu("0-02:00"),
    MEDIUM_CPU_4H: medium_cpu("0-04:00"),
    MEDIUM_CPU_8H: medium_cpu("0-08:00"),
    MEDIUM_CPU_16H: medium_cpu("0-16:00"),
    MEDIUM_CPU_24H: medium_cpu("0-24:00"),
    DEFAULT_GPU_2H: default_gpu("0-02:00"),
    DEFAULT_GPU_4H: default_gpu("0-04:00"),
    DEFAULT_GPU_8H: default_gpu("0-08:00"),
    DEFAULT_GPU_12H: default_gpu("0-12:00"),
    DEFAULT_GPU_16H: default_gpu("0-16:00"),
    DEFAULT_GPU_24H: default_gpu("0-24:00"),
    DEFAULT_GPU_36H: default_gpu("1-12:00"),
    NARVAL_GPU_2H: narval_gpu("0-02:00"),
    NARVAL_GPU_4H: narval_gpu("0-04:00"),
    NARVAL_GPU_8H: narval_gpu("0-08:00"),
    NARVAL_GPU_16H: narval_gpu("0-16:00"),
    NARVAL_GPU_24H: narval_gpu("0-24:00"),
    SMALL_GPU_1H: small_gpu("0-01:00"),
    SMALL_GPU_2H: small_gpu("0-02:00"),
    SMALL_GPU_4H: small_gpu("0-04:00"),
    SMALL_GPU_8H: small_gpu("0-08:00"),
    SMALL_GPU_12H: small_gpu("0-12:00"),
    SMALL_GPU_16H: small_gpu("0-16:00"),
    LARGE_GPU_1H: large_gpu("0-01:10"),
    LARGE_GPU_2H: large_gpu("0-02:00"),
    LARGE_GPU_2HALFH: large_gpu("0-02:30"),
    LARGE_GPU_4H: large_gpu("0-04:00"),
    LARGE_GPU_6H: large_gpu("0-06:00"),
    LARGE_GPU_8H: large_gpu("0-08:00"),
    LARGE_GPU_12H: large_gpu("0-12:00"),
    LARGE_GPU_16H: large_gpu("0-16:00"),
    LARGE_GPU_24H: large_gpu("0-24:00"),
    LARGE_GPU_36H: large_gpu("1-12:00"),
    LARGE_GPU_72H: large_gpu("3-00:00"),
}


class SlurmConfigIssue(ValueError):
    pass
