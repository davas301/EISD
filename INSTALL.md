# EISD
Install manual

To install EISD, one must first install the dependencies above. This is most
easily done with the "pip" tool, which can be installed by following the
instructions at https://pip.pypa.io/en/stable/installing/. 
Then type on the command line:

    $ pip install numpy
    $ pip install sklearn
    $ pip install biopython

Next, clone the git repository::

    $ git clone https://github.com/davas301/EISD.git

Now navigate to the downloaded directory and run the setup tool::

    $ cd eisd/
    $ python setup.py

The EISD modules can now be imported into any local python program.

Users may also want to install SHIFTX2 for chemical shift prediction, whose download and installation
instructions can be found at http://www.shiftx2.ca.


