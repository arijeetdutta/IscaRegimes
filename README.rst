====
IscaRegimes
====
|Github|

About
----

IscaRegimes is a versatile toolkit designed to extract dynamical regimes from **Isca** (https://doi.org/10.5194/gmd-11-843-2018) simulations; however, it can also be applied to reanalysis and CMIP datasets. 

This toolkit is primarily based on the **NEMI** algorithm developed by Maike Sonnewald (https://github.com/CompClimate/NEMI/), a flexible unsupervised learning framework designed to find dominant balances in geospatial data. Leveraging this framework, the toolkit introduces a novel entropy-based approach for uncertainty quantification.

It also utilises dominant balance identification code from Bryan Kaiser (https://github.com/bekaiser/dominant_balance) which is an objective way to label dynamical regimes.

An alternative package, **seaLevelRegimes** (https://github.com/CompClimate/seaLevelRegimes/), is more appropriate for handling large, memory-intensive datasets and provides functionality for batch job submission.

.. |Github| image:: https://img.shields.io/badge/GitHub-arijeetdutta%2FIscaRegimes-blue.svg?style=flat
   :target: https://github.com/arijeetdutta/IscaRegimes   