# Experiments

The synthetic data experiments are found in `synthetic-data-test.ipynb` (Barabasi-Albert graphs) and `synthetic-data-test-er.ipynb` (Erdos-Renyi graphs).

The experiments on the COIL dataset are found in `coil.iypnb`, the data for which is available [here](https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php).

The experiments on E-MTAB-2805 are found in `cell-cycle.ipynb`.  The data is available [here](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-2805).  We used a subset of the genes, first used by scBiGLasso [here](https://github.com/luisacutillo78/Scalable_Bigraphical_Lasso/blob/main/CCdata/Nmythosis.txt)

The data should hopefully be contained as part of the supplementary material, unless it was too large for us to upload, in which case the directory structure we used was:

```
/data
---/coil-20-proc
------obj1__0.png
------obj1__1.png
------[et cetera]
---/E-MTAB-2805
------G1_singlecells_counts.txt
------G2M_singlecells_counts.txt
------Nmythosis.txt
------S_singlecells_counts.txt
```

# Algorithm

Our algorithm is found in `mean_wrapper.py`, although that implementation has a few unneccessary bells and whistles to cope with GmGM's `Dataset` class as input.  `test-theory.ipynb` contains the code we used to prototype the algorithm, and hence is a simpler reference implementation if one wishes to delve into the details.

The code we used to prototype `mean_wrapper.py`, `test-mean-wrapper.ipynb`, has been included for completeness, although it is likely not of much use to anyone.

# Packages

The versions of the dependencies used for our algorithm and our tests are given in the `print-versions.ipynb` file.