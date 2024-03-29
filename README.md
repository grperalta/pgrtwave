# pgrtwave

### Petrov-Galerkin-Raviart-Thomas (PGRT) FEM for Optimal Control of the Wave Equation

> FILE DESCRIPTIONS

``pgrtwave.py``
This Python module approximates a distributed optimal control for the wave equation
written in pressure-velocity formulation using mixed and hybrid PGRT FEM.

``ocp_sample.py``
An example of an optimal control problem for the wave equation using the PGRT FEM.
Output written in ``iterhist/ocp_sample.txt``. (Example 1)

``mfem_eoc.py``
Computes the experimental orders of convergence in the mixed method. Output written 
in ``iterhist/mfem_eoc.txt``. (Example 2)

``mfem_eoc_quasiunimesh.py``
Computes the experimental orders of convergence in the mixed method with an initial
quasi-uniform mesh and successively refined by bisection. Output written 
in ``iterhist/mfem_eoc_quasiunimesh.txt``.

``hfem_eoc.py``
Computes the experimental orders of convergence in the hybrid method. 
Output written in ``iterhist/hfem_eoc.txt``.(Example 3)

``hfem_condno.py``
Prints the condition numbers of the system matrix in the hybrid formulation with
decreasing penalty parameters. Output written in ``iterhist/hfem_condno.txt``.

``hfem_eoc_penalty.py``
Computes the experimental orders of convergence in the hybrid method with decreasing
penalty parameters. Output written in ``iterhist/hfem_eoc_penalty.txt``.

``strcolor.py``
Module for output string coloring and highlighting.

``mesh``
Directory of npy files for the array of nodes and triangles. To be used in the script
``mfem_eoc_quasiunimesh.py``.

``iterhist``
Directory of iteration histories.

If you find these codes useful, you can cite the manuscript as:
> G. Peralta and K. Kunisch, [Mixed and hybrid Petrov–Galerkin finite element discretization for optimal control of the wave equation](https://link.springer.com/article/10.1007/s00211-021-01258-9), Numerische Mathematik 150 (2), pp. 591-627, 2022. [preprint](https://static.uni-graz.at/fileadmin/_Persoenliche_Webseite/kunisch_karl/Papers/wave.pdf).


Gilbert Peralta,
Department of Mathematics and Computer Science,
University of the Philippines Baguio,
Governor Pack Road, Baguio, Philippines 2600.
Email: grperalta@up.edu.ph
31 March 2021
