.. EISD documentation introduction file, created by

Getting Started
===============

Introduction
------------

Welcome to EISD! Short for Experimental Inferential Structure
Determination, EISD was developed in Teresa Head-Gordon's lab at
UC Berkeley with the goal of providing probability scores to ensembles of
Intrinsically Disordered Proteins (IDPs) based on their fit to experimental
data and conformational prior distributions. The theory underlying this
code is presented in [BrHe16]_.

EISD currently allows one to determine probabilities of ensembles using
a number of different experimental data types and priors, as well as to
optimize an ensemble by finding the subset of structures with the highest
probability.

Dependencies
------------

EISD relies on a number of Python libraries and external programs, listed
below:

* **SciPy** [JoOP01]_

* **Biopython** [CACC09]_

* **SHIFTX2** [BLGW11]_


Installation
------------

To install EISD, one must first install the dependencies above. This is most
easily done with the :code:`pip` tool, which can be installed by following the
instructions `here`_. Then type on the command line::

$ pip install numpy
$ pip install sklearn
$ pip install biopython

.. _here: https://pip.pypa.io/en/stable/installing/

Next, clone the git repository::

$ git clone https://github.com/davas301/eisd.git

Now navigate to the downloaded directory and run the setup tool::

$ cd eisd/
$ python setup.py

The EISD modules can now be imported into any local python program.

Users may also want to install **SHIFTX2** for chemical shift prediction, whose download and installation
instructions can be found on `their website`_

.. _their website: "http://www.shiftx2.ca/"