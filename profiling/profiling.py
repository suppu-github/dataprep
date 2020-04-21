"""Profiling

Usage:
  profiling --data=<dataset> --row=<row> --col=<col> --mem=<mem> --cpu=<cpu> --mode=<mode> [--partition=<part>] (pandas|dataprep)

Options:
  -h --help    Show this screen.
"""
import logging
from logging import getLogger
from pathlib import Path
from tempfile import TemporaryDirectory
from time import time
from typing import Optional
from gc import collect

import numpy as np
import pandas as pd
import dask.dataframe as dd
from contexttimer import Timer
from docopt import docopt
import missingno


logger = getLogger(__name__)
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("[{asctime} {levelname}] {name}: {message}", style="{")
ch.setFormatter(formatter)
logger.addHandler(ch)


def main() -> None:
    args = docopt(__doc__)
    print(args)

    dataset = args["--data"]
    N, M = int(args["--row"]), int(args["--col"])
    mode = args["--mode"]
    partition = args.get("--partition")

    df = create_dataset(dataset, N)
    df.reset_index(inplace=True)
    data_in_mem = df.memory_usage(deep=True).sum()

    fname = f"{dataset}_{N}_{M}.parquet"
    if partition is not None:
        df["index"] = df["index"] % int(partition)
        df.to_parquet(fname, partition_cols=["index"])
    else:
        df.to_parquet(fname)

    # release the memory
    del df
    collect()

    logger.info("Dataset dumped")

    results = []

    if args["pandas"]:
        logging.info("Benchmarking Pandas-Profiling")
        from pandas_profiling import ProfileReport

        with TemporaryDirectory() as tdir:
            with Timer() as timer:
                df = pd.read_parquet(fname)
                profile = ProfileReport(
                    df,
                    title="Pandas Profiling Report",
                    config_file=f"{Path(__file__).parent}/pandas-profiling-config.yaml",
                )
                profile.to_file(output_file=f"{Path(__file__).parent}/a_report.html")

        print(f"Pandas Profiling Elapsed: {timer.elapsed}s")
        results.append(
            {
                "Mem": args["--mem"],
                "CPU": args["--cpu"],
                "Library": "pandas-profiling",
                "Dataset": dataset,
                "Row": N,
                "Col": M,
                "MSize": data_in_mem,
                "Elapsed": timer.elapsed,
            }
        )
    else:
        from dataprep.eda.missing import plot_missing as plot_missing_new
        from dataprep.eda.missing2 import plot_missing as plot_missing_old

        with TemporaryDirectory() as tdir:
            with Timer() as timer:
                logger.info(f"Computing plot_missing_new")
                if mode == "dask":
                    df = dd.read_parquet(fname)
                elif mode == "pandas":
                    df = pd.read_parquet(fname)

                plot_missing_new(df, bins=100).save(f"{tdir}/report3.html")
        print(f"missing_new Elapsed: {timer.elapsed}s")
        results.append(
            {
                "Mem": args["--mem"],
                "CPU": args["--cpu"],
                "Mode": mode,
                "Dataset": dataset,
                "Partition": partition,
                "Row": N,
                "Col": M,
                "MSize": data_in_mem,
                "Elapsed": timer.elapsed,
                "Func": "new",
            }
        )

        with TemporaryDirectory() as tdir:
            with Timer() as timer:
                logger.info(f"Computing plot_missing_old")
                if mode == "dask":
                    df = dd.read_parquet(fname)
                elif mode == "pandas":
                    df = pd.read_parquet(fname)

                plot_missing_old(df, bins=100).save(f"{tdir}/report3.html")

        print(f"missing_old Elapsed: {timer.elapsed}s")
        results.append(
            {
                "Mem": args["--mem"],
                "CPU": args["--cpu"],
                "Mode": mode,
                "Dataset": dataset,
                "Partition": partition,
                "Row": N,
                "Col": M,
                "MSize": data_in_mem,
                "Elapsed": timer.elapsed,
                "Func": "old",
            }
        )

        with TemporaryDirectory() as tdir:
            with Timer() as timer:
                logger.info(f"Computing plot_missing_new")
                df = pd.read_parquet(fname)
                missingno.matrix(df)
                missingno.heatmap(df)
                missingno.bar(df)

        print(f"Missingno Elapsed: {timer.elapsed}s")
        results.append(
            {
                "Mem": args["--mem"],
                "CPU": args["--cpu"],
                "Mode": mode,
                "Dataset": dataset,
                "Partition": partition,
                "Row": N,
                "Col": M,
                "MSize": data_in_mem,
                "Elapsed": timer.elapsed,
                "Func": "missingno",
            }
        )

        print(dumps(results, cls=NumpyEncoder))


def create_dataset(dataset: str, n: int) -> pd.DataFrame:

    folder = Path(__file__).parent
    df = pd.read_parquet(f"{folder/dataset}.pq")

    logger.info(f"Original DataFrame shape: {df.shape}")

    rep = int(np.ceil(n / len(df)))

    df = pd.concat([df] * rep)

    df = df.sample(frac=1).reset_index(drop=True)[:n]

    logger.info(f"new DataFrame shape: {df.shape}")

    return df


if __name__ == "__main__":
    main()
