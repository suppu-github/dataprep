import sys
from time import time
from json import dumps as jdumps
from pathlib import Path

PWD = Path(__file__).resolve().parent.parent


class Benchmark:
    dpath: Path

    def __init__(self):
        self.dpath = PWD / "data" / sys.argv[1]

    def run(self) -> None:
        then = time()
        self.bench()
        elapsed = time() - then
        result = {"name": self.__class__.__name__, "elapsed": elapsed}
        print(jdumps(result))

    def bench(self) -> None:
        pass
