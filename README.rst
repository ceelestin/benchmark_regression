Benchmark for regression methods
====================================
|Build Status| |Python 3.6+|

.. warning::
    This benchmark is under development and it only run with a dev version of
    benchopt, from this PR: https://github.com/benchopt/benchopt/pull/511


Benchopt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
This benchmark is dedicated to **tabular regression methods**:


$$\\min_{w} f(X, w)$$


where $n$ (or ``n_samples``) stands for the number of samples, $p$ (or ``n_features``) stands for the number of features and


$$X \\in \\mathbb{R}^{n \\times p} \\ , \\quad w \\in \\mathbb{R}^p$$


Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install git+https://benchopt/benchopt/
   $ git clone https://github.com/ceelestin/benchmark_regression
   $ benchopt install
   $ benchopt run benchmark_regression

To parallelize the code on several CPU-cores, specify the number after the -j option, e.g. 256.
To run only 10 seeds of the code, run the config_learning_short config. To run the full 1,000 seeds, run the config_learning_full config.

.. code-block::

	$ benchopt run benchmark_regression --no-timeout -j 256 --config config_learning_full.yml


Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/ceelestin/benchmark_regression/workflows/Tests/badge.svg
   :target: https://github.com/ceelestin/benchmark_regression/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
