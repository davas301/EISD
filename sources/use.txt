.. EISD use documentation file

Use
===

Overview
--------

Given an ensemble of structures and a set of NMR experimental data, EISD
determines a Bayesian probability that the ensemble fits the experimental data.
This program embeds this probability calculation in a Simulated Annealing (SA)
procedure that allows users to find the optimal subset of structures from
an initial reservoir. At each step of the SA procedure, the current subset
is perturbed by randomly swapping a structure with one from the reservoir and
then determining the EISD probability of the perturbed subset. The difference
in EISD probabilities between the original and perturbed subsets the determines
the Metropolis acceptance probability for the perturbation, with temperature
specified by a SA cooling schedule. This procedure begins by choosing a random
subset from the reservoir and terminates after a specified number of
attempted swap iterations.

.. _input-file:

Input File
----------

In order to run this EISD optimization program from the command line, one must
given the path to an input file as the first argument::

$ eisdrun.py <input_file>

This input file is a whitespace-delimited text file that contains all
user-specified options. Each option is specified on a single line with a
keyword and value separated by whitespace. Blank lines, lines not containing
a keyword or lines starting with :code:`#` are ignored::

    KEYWORD1    value1
    KEYWORD2    value2
    # This is a comment

    KEYWORD3    value3


The table below describes the possbile keywords and value types. Values
in parentheses are the only options for those keywords. Scrool to the
right to see which keywords are required and the default values for those
that are not.

+-----------+-----------------------------------------------------+----------------------+---------+
| Keyword   | Value                                               | Required?            | Default |
+===========+=====================================================+======================+=========+
| PDBDIR    | Path to directory containing structure reservoir    | Always               |         |
+-----------+-----------------------------------------------------+----------------------+---------+
| SUB_SIZE  | Desired number of structures in output subset       | Always               |         |
+-----------+-----------------------------------------------------+----------------------+---------+
| N_ITER    | Number of SA iterations to perform                  | Recommended          | 50,000  |
+-----------+-----------------------------------------------------+----------------------+---------+
| SAVE_FILE | Path to file to file names in optimal subset        | Always               |         |
+-----------+-----------------------------------------------------+----------------------+---------+
| STATS_FILE| Path to file to save run statistics in              | Recommended          |std_out  |
+-----------+-----------------------------------------------------+----------------------+---------+
| USE_JCOUP | (true/'false) Use J coupling experimetal data?      | If USE_SHIFT is false|false    |
+-----------+-----------------------------------------------------+----------------------+---------+
| JCOUP_PATH| Path to J coupling data                             | If USE_JCOUP is true |         |
+-----------+-----------------------------------------------------+----------------------+---------+
| USE_SHIFT | (true/false) Use chemical shift  experimental data? | If USE_SHIFT is false|false    |
+-----------+-----------------------------------------------------+----------------------+---------+
| SHIFT_PATH| Path to chemical shift data                         | If USE_SHIFT is true'|         |
+-----------+-----------------------------------------------------+----------------------+---------+
| COOL_SCHED| (gaussian/linear) Type of SA cooling schedule to use| No                   |gaussian |
+-----------+-----------------------------------------------------+----------------------+---------+
| COOL_TO   | Starting temperature of cooling schedule            | No                   |0.1      |
+-----------+-----------------------------------------------------+----------------------+---------+
| COOL_SCALE| Scale of cooling scedule                            | No                   |2        |
+-----------+-----------------------------------------------------+----------------------+---------+
| PRIOR     | (uniform) Type of prior to use                      | No                   |uniform  |
+-----------+-----------------------------------------------------+----------------------+---------+
|PRIOR_UNI_M| Number of ensembles in uniform prior state space    | No                   |10       |
+-----------+-----------------------------------------------------+----------------------+---------+
| RUN_SHIFTX| (true/false) Run shiftx on every structure?         | No                   |false    |
+-----------+-----------------------------------------------------+----------------------+---------+
| SHIFT_EXE | Path to SHIFTX executable                           | If RUN_SHIFTX is true|         |
+-----------+-----------------------------------------------------+----------------------+---------+

An example input file can be found in :code:`$EISD_HOME/test/ab42_example.inp`.

Experimental Data Files
-----------------------

Experimental data files that are input into EISD must have a specific format,
which is the same as that used in the ENSEMBLE program. The format for each
experimental data point is described below, with examples.

Chemical Shift
^^^^^^^^^^^^^^
::

    Residue     Atom    Chemical Shift      Error

J-Coupling
^^^^^^^^^^
::

    Res1    Atom1   Res2    Atom2   Res3    Atom3   Res4    Atom4   J   Err_Low   Err_Up


Example files can be found in :code:`$EISD_HOME/test/`


















































