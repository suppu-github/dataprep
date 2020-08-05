import logging
import sys
from functools import lru_cache
from gc import collect
from json import JSONDecodeError
from json import dumps as jdumps
from json import loads as jloads
from logging import getLogger
from os import listdir
from pathlib import Path
from time import sleep
from typing import Any, Optional, Tuple

import docker
import numpy as np
import pandas as pd
from docker.models.containers import Container
from sklearn.model_selection import ParameterGrid

from numpyencoder import NumpyEncoder

logger = getLogger(__name__)
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("[{asctime} {levelname}] {name}: {message}", style="{")
ch.setFormatter(formatter)
logger.addHandler(ch)

PWD = Path(__file__).resolve().parent
G = 1024 * 1024 * 1024

RESOURCES = {"memory": 60 * G, "cpu": 8}


def grid() -> ParameterGrid:
    fnames = []
    for fname in listdir(PWD / "benchmarks"):
        if fname.endswith(".py") and fname != "lib.py":
            fnames.append(fname)

    return ParameterGrid(
        {
            "dataset": ["adult"],
            "memory": [1 * G, 2 * G, 4 * G, 8 * G],
            "cpu": [8],
            "fname": fnames,
            "nrow": [50000, 500000, 5000000, 10000000],
            "ncol": [12],
            "partition": [16],
        }
    )

    # return ParameterGrid(
    #     {
    #         "dataset": ["titanic"],
    #         "memory": [8 * G],
    #         "cpu": [8],
    #         "fname": fnames,
    #         "nrow": [10000],
    #         "ncol": [12],
    #         "partition": [16],
    #     }
    # )


def main() -> None:
    dclient = docker.from_env()

    running_jobs = {}

    for args in grid():
        mem = args["memory"]
        cpu = args["cpu"]
        fname = args["fname"]

        dpath, size_in_mem = create_dataset(
            args["dataset"], args["nrow"], args["ncol"], args["partition"]
        )
        args["mem_size"] = size_in_mem

        while RESOURCES["memory"] < mem or RESOURCES["cpu"] < cpu:
            deletions = []
            # resources is not enough, wait for something to be finished
            for job, this_args in running_jobs.items():
                job.reload()
                if job.status == "exited":
                    ret = extract_result(job)
                    if ret is not None:
                        print(jdumps({**this_args, **ret}, cls=NumpyEncoder))
                    else:
                        print(
                            jdumps({**this_args, "status": "failed"}, cls=NumpyEncoder)
                        )
                    sys.stdout.flush()

                    this_mem = this_args["memory"]
                    this_cpu = this_args["cpu"]
                    RESOURCES["memory"] += this_mem
                    RESOURCES["cpu"] += this_cpu

                    deletions.append(job)

            for job in deletions:
                del running_jobs[job]
                job.remove()

            if not deletions:
                sleep(1)

        logger.info(f"Running: {args}, Current Resources: {RESOURCES}")

        RESOURCES["memory"] -= mem
        RESOURCES["cpu"] -= cpu

        job = dclient.containers.run(
            "wooya/dataprep-profiling",
            f"python benchmarks/{fname} {dpath}",
            detach=True,
            volumes={
                str(PWD.parent / "dataprep"): {
                    "bind": "/workdir/dataprep",
                    "mode": "ro",
                },
                str(PWD / "benchmarks"): {"bind": "/workdir/benchmarks", "mode": "ro",},
                str(PWD / "data"): {"bind": "/workdir/data", "mode": "ro"},
            },
            environment=["PYTHONPATH=/workdir:$PYTHONPATH"],
            mem_limit=mem,
            cpu_period=100000,
            cpu_quota=cpu * 100000,
        )
        running_jobs[job] = args

    # Wait for the remaining jobs to be finished

    while running_jobs:
        deletions = []
        for job, this_args in running_jobs.items():
            job.reload()
            if job.status == "exited":
                ret = extract_result(job)
                if ret is not None:
                    print(jdumps({**this_args, **ret}, cls=NumpyEncoder))
                else:
                    print(jdumps({**this_args, "status": "failed"}, cls=NumpyEncoder))
                sys.stdout.flush()
                deletions.append(job)

        for job in deletions:
            del running_jobs[job]
            job.remove()

        if not deletions:
            sleep(1)


def extract_result(job: Container) -> Any:
    logs = job.logs()
    logs = logs.strip()
    logger.info(f"Log from benchmark {logs.decode()}")

    logs = logs.splitlines()

    ret = None
    for line in reversed(logs):
        try:
            ret = jloads(line)
            break
        except JSONDecodeError:
            pass

    return ret


@lru_cache(maxsize=128)
def create_dataset(
    dataset: Path,
    nrow: int,
    ncol: int,
    partition: Optional[int] = None,
    skip: bool = True,
) -> Tuple[str, float]:
    fname = f"{dataset}_{nrow}_{ncol}_{partition}.pq"
    fpath = PWD / "data" / fname

    if skip and fpath.exists():
        size_in_mem = pd.read_parquet(fpath).memory_usage(deep=True).sum()

        return fname, size_in_mem

    df = pd.read_parquet(f"{PWD/'data'/dataset}.pq")

    logger.info(f"Original DataFrame shape: {df.shape}")

    rep = (int(np.ceil(nrow / len(df))), int(np.ceil(ncol / len(df.columns))))

    df = pd.concat([df] * rep[0])
    df = pd.concat([df] * rep[1], axis=1)
    df = df.sample(frac=1).reset_index(drop=True).iloc[:nrow, :ncol]

    logger.info(f"new DataFrame shape: {df.shape}")

    df.reset_index(inplace=True)
    size_in_mem = df.memory_usage(deep=True).sum()

    if partition is not None:
        df["index"] = df["index"] % int(partition)
        df.to_parquet(fpath, partition_cols=["index"])
    else:
        df.to_parquet(fpath)

    # release the memory
    del df
    collect()

    return fname, size_in_mem


if __name__ == "__main__":
    main()
