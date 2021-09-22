# Possibility aggregation

In order to compute infimum and supremum of two possibility distributions `pi_1` and `pi_2`:
1. Make necessary imports
  ```python
  import numpy as np
  from possibilities import PossibiltyDistribution as PD
  ```
2. Define distributions `pi_1` and `pi_2`
  ```python
  pi_1 = PD([1., 1., 0.5])
  pi_2 = PD([1., 0.5, 1.])
  ```
3. Use operators `^` and `|` to compute infimum and supremum respectevly
  ```python
  sup = pi_1 | pi_2
  inf = pi_1 ^ pi_2
  ```
Look into [example.py](https://github.com/NeuroFuzzyLab/possibility_aggregation/blob/main/example.py) to see more detailed example.
## Citing

### BibTeX
If you use the code in this package, please consider citing:

```bibtex
@misc{Zubyuk21,
  author = {Andrey Zubyuk, Egor Fadeev},
  title = {Aggregation Operators for Comparative Possibility Distributions and Their Role in Group Decision Making},
  year = {2021},
  publisher = {Atlantis Press},
  journal = {Joint Proc. of IFSA-EUSFLAT 21},
  doi = {10.2991/asum.k.210827.082},
}
```
<!---
[![DOI](https://zenodo.org/badge/168799526.svg)](https://zenodo.org/badge/latestdoi/168799526)
-->
