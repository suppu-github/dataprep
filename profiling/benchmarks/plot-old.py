from lib import Benchmark


class PlotOld(Benchmark):
    def bench(self) -> None:
        from dataprep.eda.basic import plot
        from tempfile import TemporaryDirectory
        import dask.dataframe as dd
        import pandas as pd

        with TemporaryDirectory() as tdir:
            df = dd.read_parquet(self.dpath)

            plot(df).save(f"{tdir}/report.html")


if __name__ == "__main__":
    PlotOld().run()
