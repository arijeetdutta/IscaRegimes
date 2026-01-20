====
IscaRegimes
====
|Github|

About
----

IscaRegimes is a versatile toolkit designed to extract dynamical regimes from **Isca** (https://doi.org/10.5194/gmd-11-843-2018) simulations; however, it can also be applied to reanalysis and CMIP datasets. 

Acknowledgement
----

This toolkit is primarily based on the **NEMI** algorithm developed by Maike Sonnewald (https://github.com/CompClimate/NEMI/), a flexible unsupervised learning framework designed to find dominant balances in geospatial data.

It also utilizes the dominant balance identification code from Bryan Kaiser (https://github.com/bekaiser/dominant_balance), which is an objective method for labeling dynamical regimes.

.. |Github| image:: https://img.shields.io/badge/GitHub-arijeetdutta%2FIscaRegimes-blue.svg?style=flat
   :target: https://github.com/arijeetdutta/IscaRegimes   

Installation
----

Python 3.8 or greater is required

1. Clone the repository
2. (Optional) create a virtual environment
3. Go to the repository root and install the package:

``pip install -e .[full]``

Example
----

For a quick example check the Jupyter notebook ``check.ipynb`` in ``test`` directory
