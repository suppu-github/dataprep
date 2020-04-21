FROM python:3.7

RUN apt update && apt install -y libsnappy-dev
RUN pip install pandas_profiling docopt contexttimer pandas numpy fastparquet python-snappy missingno
RUN pip install git+https://github.com/sfu-db/dataprep@perf
