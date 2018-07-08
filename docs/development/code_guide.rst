.. _code-development-guidelines:

***************************
Code Development Guidelines
***************************

This document describes the coding requirements and guidelines to be
followed during the development of PlasmaPy and affiliated packages.

Code written for PlasmaPy must be compatible with Python 3.6 and
later. Python 2 is not supported by PlasmaPy.

Obtaining PlasmaPy source code
==============================

After creating your GitHub account, go to the [main
repository](https://github.com/PlasmaPy/PlasmaPy) and **fork a copy of
PlasmaPy to your account**.

To access Git commands on Windows, try `Git Bash
<https://git-scm.com/downloads>`_.

Next you must **clone your fork to your computer**.  Go to the
directory that will host your PlasmaPy directory, and run one of the
following commands (after changing *your-username* to your username).
If you would like to use HTTPS (which is the default and easier to set
up), then run:

.. code-block:: bash

  git clone https://github.com/your-username/PlasmaPy.git

SSH is a more secure option, but requires you to [set up an SSH
key](https://help.github.com/articles/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent/)
beforehand.  The equivalent SSH command is:

.. code-block:: bash

    git clone git@github.com:your-username/PlasmaPy.git
    cd PlasmaPy

After cloning, we must tell git where the development version of
PlasmaPy is by running:

.. code-block:: bash

  git remote add upstream git://github.com/PlasmaPy/PlasmaPy.git

To check on which remotes exist, run `git remote -v`.  You should get
something like this:

.. code-block:: bash

  origin	git@github.com:namurphy/PlasmaPy.git (fetch)
  origin	git@github.com:namurphy/PlasmaPy.git (push)
  upstream	git@github.com:PlasmaPy/PlasmaPy.git (fetch)
  upstream	git@github.com:PlasmaPy/PlasmaPy.git (push)

Setting up an environment for development
=========================================

Setup procedures for the two most popular virtual environments, conda
and virtualenv, are listed below.

Conda
-----

To set up a development environment for PlasmaPy, `the Anaconda
distribution <https://www.anaconda.com/download/>`_ is strongly
recommended.

Activate Anaconda
~~~~~~~~~~~~~~~~~

After installing Anaconda, launch any conda environment. By default,
conda installs a `root` environment, which you should be able to
activate via

.. code-block:: bash

  source /home/user/anaconda3/bin/activate root

where `/home/user/anaconda3/` can be swapped to wherever your anaconda
installation resides.

On `newer versions of Anaconda <https://conda.io/docs/release-notes
.html#recommended-change-to-enable-conda-in-your-shell>`_ the
recommended activation process has changed to:

.. code-block:: bash

  . /home/user/anaconda3/etc/profile.d/conda.sh
  conda activate

.. note::
    On Windows, the way to do this is via running `Anaconda Prompt` from the
    Start Menu. `Git Bash` may also work if you have added Anaconda to `PATH`.

Create your environment
~~~~~~~~~~~~~~~~~~~~~~~

Having activated Anaconda, enter PlasmaPy's repository root directory and
create an environment with our suggested packages by executing the following:

.. code-block:: bash

    conda env create -f requirements/environment.yml

You may now enter the environment via

.. code-block:: bash

    source activate plasmapy

.. note::

    On Windows, skip the `source` part of the previous command.

In newer Conda versions, the command to run is

.. code-block:: bash

    conda activate plasmapy

Virtualenv
----------

Create a directory for holding the PlasmaPy repository, move into it
and create the virtual environment

.. code-block:: bash

    virtualenv -p python3 .

You may need to make sure that this directory's path doesn't contain
any spaces, otherwise virtualenv may throw an error.

Your virtual environment should now be created. If you run `ls` you
will notice that virtualenv has created a number of subdirectories:
`bin/`, `lib/`, and `include/`. This is why we're not creating the
virtualenv within the repository itself - so as to not pollute it. To
activate the virtualenv you will run:

.. code-block:: bash

    source ./bin/activate

You should now see that your shell session is prepended with
(plasmapy), like so:

.. code-block:: bash

    (plasmapy) user@name:~/programming/plasmapy$

This indicates that the virtualenv is running. Congratulations!  When
your're done working on PlasmaPy, you can deactivate the virtualenv by
running

.. code-block:: bash

    source deactivate

Now that you have plasmapy on your local computer and you have a
virtual environment, you will want to "install" this development
version of PlasmaPy along with its dependencies. Start by activating
your virtual environment. Next you want install the PlasmaPy
dependencies. One way to do this is to do

.. code-block:: bash

    (plasmapy) user@name:~/programming/plasmapy$ pip install -r requirements/environment.txt

Next, setup the development version of PlasmaPy which you just cloned
by moving into the root directory of the cloned repo and running the
setup.py script there:

.. code-block:: bash

    (plasmapy) user@name:~/programming/plasmapy/PlasmaPy$ pip install -e .


You should now be all set to run development versions of PlasmaPy
modules via `import PlasmaPy` in your test scripts!

Running anaconda with virtualenv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are running the Anaconda suite and want to use virtualenv to
setup your virtual environment, you will have to let the system know
where the Python interpreter can be found. On Linux this is done with
(for example, assuming having installed Anaconda into `~/anaconda3`):

.. code-block:: bash

    export LD_LIBRARY_PATH="$HOME/anaconda3/lib/"

Exporting the library path to the dynamic linker will only last for
the duration of the current shell session.

You will have to add the python library directory to LD_LIBRARY_PATH,
as described in a previous step, prior to activating the virtualenv
for every new shell session.

Installing your own dev version
===============================
To be able to import PlasmaPy from your source version, enter the repository
root and use one of

.. code-block:: bash

  python setup.py develop
  pip install -e .

.. note::

    If you are not working within a virtual environment, this may end in a
    permission error - this can be avoided via also adding the `--user`
    flag. But seriously, use a virtual environment and spare yourself the trouble.

Either one of these commands will create a soft link to your cloned repository.
Any changes in Python code you make there will be there when you `import
plasmapy` from an interactive session. The exception is Cython code.

Working with Cython code
========================

.. note::
     
    We are still figuring this part out. Contributions are very welcome!

Cython poses a whole new set of issues for code development.  When writing
Cython code, you need to recompile your current files for your changes to take
effect. This is accomplished, to the best of our current understanding , via
`python setup.py build_ext -i`, as recommended by the Cython development guide
and as supported by the Astropy package template in `astropy_helpers`. This
places compiled `.c` or `.cpp` sources and `.so` compiled libraries in the same
location as your original `.pxd` and `.py` sources - this plays well with import
statements.

Note that `python setup.py install` or `pip install` will also build extensions.

For now, the `.c` and `.cpp` sources are not included with the source code, meaning
that they are compiled on installation. This solution is what we got working
for distribution, but may change in the future.

The safest bet for running tests and checking documentation is using `python
setup.py test` and `python setup.py build_docs` provided by `astropy_helpers`.
These commands copy the entire code base into a temporary directory for
isolation and build all Cython extensions there before running tests and
documenation builds.

Coding Style
============

* PlasmaPy follows the `PEP8 Style Guide for Python Code
  <http://www.python.org/dev/peps/pep-0008/>`_.  This style choice
  helps ensure that the code will be consistent and readable.

  * The PEP 8 Speaks integration on GitHub will comment when there are
    any departures from the PEP 8 style guide.

  * PEP 8 compliance may be checked locally using
    [pycodestyle](http://pycodestyle.pycqa.org/en/latest/).

  * Line lengths should be chosen to maximize the readability and
    elegance of the code.  The maximum line length for Python code in
    PlasmaPy is 99 characters.

  * Docstrings and comments should generally be limited to
    72 characters.

* Follow the existing coding style within a subpackage.

* Use standard abbreviations for imported packages when possible, such
  as ``import numpy as np``, ``import matplotlib as mpl``, ``import
  matplotlib.pyplot as plt``, and ``import astropy.units as u``.

* ``__init__.py`` files for modules should not contain any significant
  implementation code, but it can contain a docstring describing the
  module and code related to importing the module.  Any substantial
  functionality should be put into a separate file.

* Use absolute imports, such as
  ``from plasmapy.mathematics import Fermi_integral``,
  rather than relative imports such as
  ``from ..mathematics import Fermi_integral``.

* For multiline imports, instead of using ``\`` to break lines, wrap the
  imported names in ``()`` parentheses and use consistent whitespace.

* There should be at most one pun per 1284 lines of code.

Branches, commits, and pull requests
====================================

Before making any changes, it is prudent to update your local
repository with the most recent changes from the development
repository:

.. code-block:: bash

  git fetch upstream

Changes to PlasmaPy should be made using branches.  It is usually best
to avoid making changes on your master branch so that it can be kept
consistent with the upstream repository.  Instead we can create a new
branch for the specific feature that you would like to work on:

.. code-block:: bash

  git branch *your-new-feature*

Descriptive branch names such as `grad-shafranov` or
`adding-eigenfunction-poetry` are helpful, while vague names like
`edits` are considered harmful.  After creating your branch locally,
let your fork of PlasmaPy know about it by running:

.. code-block:: bash

  git push --set-upstream origin *your-new-feature*

It is also useful to configure git so that only the branch you are
working on gets pushed to GitHub:

.. code-block:: bash

  git config --global push.default simple

Once you have set up your fork and created a branch, you are ready to
make edits to PlasmaPy.  Switch to your new branch by running:

.. code-block:: bash

  git checkout *your-new-feature*

Go ahead and modify files with your favorite text editor.  Be sure to
include tests and documentation with any new functionality.  We
recommend reading about `best practices for scientific computing
<https://doi.org/10.1371/journal.pbio.1001745>`_.  PlasmaPy uses the
`PEP 8 style guide for Python code
<https://www.python.org/dev/peps/pep-0008/>`_ and the `numpydoc format
for docstrings
<https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_
to maintain consistency and readability.  New contributors should not
worry too much about precisely matching these styles when first
submitting a pull request, as the `PEP8 Speaks
<http://pep8speaks.com/>`_ GitHub integration will check pull requests
for PEP 8 compatibility, and further changes to the style can be
suggested during code review.

You may periodically commit changes to your branch by running

.. code-block:: bash

  git add filename.py
  git commit -m "*brief description of changes*"

Committed changes may be pushed to the corresponding branch on your
GitHub fork of PlasmaPy using

.. code-block:: bash

  git push origin *your-new-feature*

or, more simply,

.. code-block:: bash

  git push

Once you have completed your changes and pushed them to the branch on
GitHub, you are ready to make a pull request.  Go to your fork of
PlasmaPy in GitHub.  Select "Compare and pull request".  Add a
descriptive title and some details about your changes.  Then select
"Create pull request".  Other contributors will then have a chance to
review the code and offer contructive suggestions.  You can continue
to edit the pull request by changing the corresponding branch on your
PlasmaPy fork on GitHub.  After a pull request is merged into the
code, you may delete the branch you created for that pull request.

Commit Messages
---------------
Good commit messages communicate context and intention to other
developers and to our future selves.  They provide insight into why we
chose a particular implementation, and help us avoid past mistakes.

Suggestions on `how to write a git commit message
<https://chris.beams.io/posts/git-commit/>`_:

* Separate subject from body with a blank line

* Limit the subject line to 50 characters

* Capitalize the subject line

* Do not end the subject line with a period

* Use the imperative mood in the subject line

* Wrap the body at 72 characters

* Use the body to explain what and why vs. how

Documentation
=============

* All public classes, methods, and functions should have docstrings
  using the numpydoc format.

* Docstrings may be checked locally using
  [pydocstyle](http://www.pydocstyle.org/en/latest/).

* These docstrings should include usage examples.

Warnings and Exceptions
=======================

* Debugging can be intensely frustrating when problems arise and the
  associated error messages do not provide useful information on the
  source of the problem.  Warnings and error messages must be helpful
  enough for new users to quickly understand any problems that arise.

* "Errors should never pass silently."  Users should be notified when
  problems arise by either issuing a warning or raising an exception.

* The exceptions raised by a method should be described in the
  method's docstring.  Documenting exceptions makes it easier for
  future developers to plan exception handling.

Units
=====

* Code within PlasmaPy must use SI units to minimize the chance of
  ambiguity, and for consistency with the recognized international
  standard.  Physical formulae and expressions should be in base SI
  units.

  * Functions should not accept floats when an Astropy Quantity is
    expected.  In particular, functions should not accept floats and
    make the assumption that the value will be in SI units.

  * A common convention among plasma physicists is to use
    electron-volts (eV) as a unit of temperature.  Strictly speaking,
    this unit corresponds not to temperature but is rather a measure
    of the thermal energy per particle.  Code within PlasmaPy must use
    the kelvin (K) as the unit of temperature to avoid unnecessary
    ambiguity.

* PlasmaPy uses the astropy.units package to give physical units to
  values.

  * All units packages available in Python presently have some
    limitations, including incompatibility with some NumPy and SciPy
    functions.  These limitations are due to issues within NumPy
    itself.  Many of these limitations are being resolved, but require
    upstream fixes.

* Dimensionless units may be used when appropriate, such as for
  certain numerical simulations.  The conventions and normalizations
  should be clearly described in docstrings.

Equations and Physical Formulae
===============================

* If a quantity has several names, then the function name should be
  the one that provides the most physical insight into what the
  quantity represents.  For example, ``gyrofrequency`` indicates
  gyration, whereas ``Larmor_frequency`` indicates that this frequency
  is somehow related to someone named Larmor.  Similarly, using
  ``omega_ce`` as a function name will make the code less readable to
  people who are unfamiliar with this particular notation.

* Physical formulae should be inputted without first evaluating all of
  the physical constants.  For example, the following line of code
  obscures information about the physics being represented:

>>> omega_ce = 1.76e7*(B/u.G)*u.rad/u.s   # doctest: +SKIP

  In contrast, the following line of code shows the exact formula
  which makes the code much more readable.

>>> omega_ce = (e * B) / (m_e * c)       # doctest: +SKIP

  The origins of numerical coefficients in formulae should be
  documented.

* Docstrings should describe the physics associated with these
  quantities in ways that are understandable to students who are
  taking their first course in plasma physics while still being useful
  to experienced plasma physicists.

* SI units that were named after a person should not be capitalized
  except at the beginning of a sentence.

* Some plasma parameters depend on more than one quantity with
  the same units.  In the following line, it is difficult to discern which
  is the electron temperature and which is the ion temperature.

  >>> ion_sound_speed(1e6*u.K, 2e6*u.K)  # doctest: +SKIP

  Remembering that "explicit is better than implicit", it is more
  readable and less prone to errors to write:

  >>> ion_sound_speed(T_i=1e6*u.K, T_e=2e6*u.K)    # doctest: +SKIP

* SI units that were named after a person should be lower case except at
  the beginning of a sentence, even if their symbol is capitalized. For
  example, kelvin is a unit while Kelvin was a scientist.


Angular Frequencies
===================

Unit conversions involving angles must be treated with care.  Angles
are dimensionless but do have units.  Angular velocity is often given
in units of radians per second, though dimensionally this is
equivalent to inverse seconds.  Astropy will treat radians
dimensionlessly when using the ``dimensionless_angles`` equivalency,
but ``dimensionless_angles`` does not account for the multiplicative
factor of ``2*pi`` that is used when converting between frequency (1 /
s) and angular frequency (rad / s).  An explicit way to do this
conversion is to set up an equivalency between cycles/s and Hz:

>>> from astropy import units as u
>>> f_ce = omega_ce.to(u.Hz, equivalencies=[(u.cy/u.s, u.Hz)])   # doctest: +SKIP

However, ``dimensionless_angles`` does work when dividing a velocity
by an angular frequency to get a length scale:

>>> d_i = (c/omega_pi).to(u.m, equivalencies=u.dimensionless_angles())    # doctest: +SKIP
