=====================
Installation and test
=====================

Holotomo works ... . To run holotomo the system should have ...

1. Add conda-forge to anaconda channels

::

    (base)$ conda config --add channels conda-forge
    (base)$ conda config --set channel_priority strict


2. Create environment with installed holotomo

::

    (base)$ conda create -n holotomo holotomo

4. Activate holotomo environment

::

    (base)$ conda activate holotomo
    

5. Test installation

::

    (holotomo)$ holotomo ...

============================
Installation for development
============================

1. Add conda-forge to anaconda channels

::

    (base)$ conda config --add channels conda-forge
    (base)$ conda config --set channel_priority strict

2. Create environment with necessary dependencies

::

    (base)$ conda create -n holotomo -c conda-forge .....


4. Activate holotomo environment

::

    (base)$ conda activate holotomo

Update
======

**holotomo** is constantly updated to include new features. To update your locally installed version

::

    (holotomo)$ cd holotomo
    (holotomo)$ git pull
    (holotomo)$ pip install .
