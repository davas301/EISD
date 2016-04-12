.. EISD use documentation file

Use
===

Calculating EISD probabilities
------------------------------

The most basic use of EISD is to determine a probability score based on
an ensemble of structure's fit to experimental data and some prior distribution on
conformations. This requires back-calculating experimental observables from the ensemble's structure
and determining the probability of the error between these values and the
true experimental value. This procedure is currently implemented for J-coupling
constants and chemical shifts. Prior probabilities are then calculated separately. EISD
only currently implements a uniform prior distribution. See [BrHe16]_ for more
details on these procedures.

Below is a tutorial on how to calculate these probabilities using EISD.
This tutorial assumes that SHIFTX2 has been run on every structure in the ensemble
and the output files are saved with the ".cs" extension

We begin by building the structures in the ensemble using :class:`structure.Structure`::

    import os
    from eisd.structure import Structure

    pdbdir = "$PATH_TO_ENSEMBLE_DIR$/"  # path to directory of PDB files

    all_pdb_paths = [os.path.join(pdbdir, f) for f in os.listdir(pdbdir) if
                     ".pdb" in f and ".cs" not in f]

    structs = [Structure(f, shiftxfile=f + ".cs") for f in all_pdb_paths]

Next we load in experimental data. The methods below (found in :mod:`readutil`)
are specific to the data files in the :code:`test` directory but can adapted
for any experimental data format, as long as it returns a dictionary with
:class:`readutil.BaseDataID` objects as keys and :code:`(observable, error)`
tuples as values::

    import readutil

    jcoup_data = readutil.get_ab42_jcoup_data()
    shift_data = readutil.get_ab42_shift_data()

Next we build our :class:`backcalc.BaseBackCalculator` objects that will
be used to approximately calculate experimental observables from structures information::

    from eisd.backcalc import JCoupBackCalc, ShiftBackCalc

    jcoup_back_calc = JCoupBackCalc()
    shift_back_calc = ShiftBackCalc()

And now we use the above to build :class:`DataEISD` objects that will calculate
probabilities::

    from eisd.eisd import DataEISD

    shift_eisd = DataEISD(shift_back_calc, shift_data)

    # don't consider back-calc error for j-coupling (error is all in nuisance parameter variability):
    jcoup_eisd = DataEISD(jcoup_back_calc, jcoup_data, no_bc_err=True)

Now build a :class:`priors.UniformPrior` object assuming there are
10 ensembles in our hypothesis space (so the prior probability of each is 0.1)::

    from eisd.priors import UniformPrior

    prior = UniformPrior(10)

Now we calculate and print the probability of the ensemble given the
experimental data and prior distribution::

    shift_logp = shift_eisd.calc_logp(structs)
    jcoup_logp = jcoup_eisd.calc_logp(structs)
    prior_logp = prior.calc_prior_logp()
    total_logp = shift_logp + jcoup_logp + prior_logp()

    outstr = "The ensemble at %s fits chemical shift data with log probability of %.3f, " \
             "J coupling data with log probability of %.3f and a conformational prior" \
              "distribution with log probability %.3f" The total log probability is then %.3f" \
              % (pdbdir, shift_logp, jcoup_logp, prior_logp, total_logp)
    print outstr


Finding optimal subsets
-----------------------

Due to EISD's size extensivity, it is ideal for finding the optimal subset of
an ensemble that maximizes the probabilities found above. This optimization
is implemented in the using a simulated annealing to swap out structures in the subset
with structures from the full ensemble.

Below is an example showing how to perform this optimization with J-coupling
and chemical shift experimental data and a Uniform Prior. Again, this assumes
that SHIFTX2 has been run for every structure in the full ensemble.

First define the directory where the full ensemble of structures is stored.
This is referred to as the "reservoir" of structures. Also define the size
of the subset that will be optimized and the number of iterations of
simulated annealing to be performed. Additionally provide the path to a  file that will
contain the names of the pdb files that make up the optimal ensemble, as well
as a file for storing optimization statistics::

    pdbdir = "$PATH_TO_ENSEMBLE_DIR$/"
    subset_size = 1000
    niter = int(1e6)

    savefile = "../output/$OPT_FILE$"
    stats_file = "../output/$STATS_FILE$"

Now build a :class:`priors.BasePrior` object and :class:`eisd.DataEISD` objects
for each set of experimental data::

    from eisd.priors import UniformPrior
    from eisd.eisd import DataEISD
    from backcalc import JCoupBackCalc, ShiftBackCalc
    import eisd.readutil

    prior = UniformPrior(niter)  # assume every iterations produces a new hypothesis ensemble

    jcoup_eisd = DataEISD(JCoupBackCalc(), eisd.readutil.get_ab42_jcoup_data(),
                          no_bc_err=True)
    shift_eisd = DataEISD(ShiftBackCalc(), eisd.readutil.get_ab42_shift_data())

    # put eisd's together in a list:
    data_eisds = [jcoup_eisd, shift_eisd]

Now define a cooling schedule for simulated annealing. Below is the default
schedule::

    def cool_sched(t):
        """
        Cooling schedule

        :param t: fraction of iterations
        :return: "Temperature" for simulated annealing
        """
        return np.exp(-(2 * t) ** 2)


Now we can build the :class:`eisd.EISDOPT` object and begin the optimization::

    from eisd.eisd import EISDOPT

    optimizer = EISDOPT(pdbdir, prior, data_eisds, savefile,
                        subsize=subset_size, verbose=True,
                        stats_file=stats_file)

    optimizer.opt(niter, cool_sched=cool_sched)











