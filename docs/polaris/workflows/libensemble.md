# libEnsemble

libEnsemble is a Python toolkit for running dynamic, portable ensembles of calculations on or across laptops, clusters, and supercomputers. libEnsemble autodetects system details and can dynamically manage resources.

Users select or provide generator and simulator functions to express their ensembles, and the generator steers the ensemble based on previous simulator results. Such functions can submit and coordinate external executables at any scale. A library of example functions is available.

## Getting libEnsemble on Polaris

libEnsemble is provided on Polaris in the **conda** module:

    module load conda
    conda activate base

See the docs for more details on using [python on Polaris](https://docs.alcf.anl.gov/polaris/data-science-workflows/python/).

<details>
  <summary>Example: creating virtual environment and updating libEnsemble</summary>

    E.g., to create a virtual environment that allows installation of
    further packages with pip:

    ```bash
    python -m venv /path/to-venv --system-site-packages
    . /path/to-venv/bin/activate
    ```

    Where /path/to-venv can be anywhere you have write access.
    For future uses just load the conda module and run the activate line.

    You can also ensure you are using the latest version of libEnsemble:

    ```bash
    pip install libensemble
    ```
</details>


## libEnsemble examples

For a very simple example of using libEnsemble see the [Simple Sine tutorial](https://libensemble.readthedocs.io/en/main/tutorials/local_sine_tutorial.html)

For an example that runs a small ensemble using a GPU application see
[the GPU app tutorial](https://libensemble.readthedocs.io/en/main/tutorials/forces_gpu_tutorial.html). The required files for the this tutorial can be found [here](https://github.com/Libensemble/libensemble/tree/develop/libensemble/tests/scaling_tests/forces). Also, see the
[video demo](https://youtu.be/Ff0dYYLQzoU).

libEnsemble also features a ``BalsamExecutor`` module for submitting apps to [Balsam](https://balsam.readthedocs.io/en/latest/).

### Code Sample

```python

import numpy as np
from libensemble import Ensemble
from libensemble.specs import SimSpecs, GenSpecs, ExitCriteria, LibeSpecs
from libensemble.executors import Executor, MPIExecutor
from libensemble.message_numbers import TASK_FAILED, WORKER_DONE
from libensemble.gen_funcs.sampling import uniform_random_sample


def run_forces(H, _, sim_specs):

    particles = str(int(H["x"][0][0]))

    exctr = Executor.executor
    task = exctr.submit(app_name="forces", app_args=particles)
    task.wait()

    try:
        data = np.loadtxt("forces.stat")
        final_energy = data[-1]
        calc_status = WORKER_DONE
    except Exception:
        final_energy = np.nan
        calc_status = TASK_FAILED

    output = np.zeros(1, dtype=sim_specs["out"])
    output["energy"] = final_energy

    return output, _, calc_status


if __name__ == "__main__":
    forces_study = Ensemble()

    exctr = MPIExecutor()
    sim_app = os.path.join(os.getcwd(), "../forces_app/forces.x")
    exctr.register_app(full_path=sim_app, app_name="forces")

    forces_study.sim_specs = SimSpecs(
        sim_f = run_forces,
        inputs = ["x"],
        out = [("energy", float)],
    )

    forces_study.gen_specs = GenSpecs(
        gen_f = uniform_random_sample,
        out = [("x", float, (1,))],
        user = {
            "lb": np.array([1000]),
            "ub": np.array([3000]),
            "gen_batch_size": 8,
        }
    )

    forces_study.libE_specs = LibeSpecs(ensemble_dir_path="/scratch")
    forces_study.exit_criteria = ExitCriteria(sim_max=800)

    results = forces_study.run()
```

## Job Submission

libEnsemble runs on the compute nodes on Polaris using either
``multiprocessing`` to colocate all processes or ``mpi4py`` to distribute processes
across nodes. Either way, applications can be launched onto separate nodes
from libEnsemble's processes.

A simple example batch script featuring four workers on one node:

```shell
    #!/bin/bash -l
    #PBS -l select=1:system=polaris
    #PBS -l walltime=00:15:00
    #PBS -l filesystems=home:grand
    #PBS -q debug
    #PBS -A <myproject>

    export MPICH_GPU_SUPPORT_ENABLED=1
    cd $PBS_O_WORKDIR
    python run_libe_forces.py --comms local --nworkers 4
```

The script can be run with:

    qsub submit_libe.sh

Or you can run an interactive session with:

    qsub -A <myproject> -l select=1 -l walltime=15:00 -lfilesystems=home:grand -qdebug -I

## Further links

Docs: <https://libensemble.readthedocs.io> <br>
GitHub: <https://github.com/Libensemble/libensemble>

