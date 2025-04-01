Benchmark for regression methods
====================================
|Build Status| |Python 3.6+|

Benchopt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
This benchmark is dedicated to **tabular regression methods**.


Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install git+https://benchopt/benchopt/
   $ git clone https://github.com/ceelestin/benchmark_regression
   $ benchopt install benchmark_regression
   $ benchopt run benchmark_regression --no-timeout -j 256 --config config_learning_full.yml

To parallelize the code on several CPU-cores, specify the number of cores after the -j option, e.g. 256.
To run only 10 seeds of the code, run the config_learning_short config. To run the full 1,000 seeds, run the config_learning_full config.

Afterwards, modify the file_name variable in learning_ranking.py by the name of the parquet output by the benchmark, and then run the learning_ranking.py file to obtain several plots describing the results of the benchmark.


Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/ceelestin/benchmark_regression/workflows/Tests/badge.svg
   :target: https://github.com/ceelestin/benchmark_regression/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
