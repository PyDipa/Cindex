# Cindex
 A recursive calculation to identify the best Composite index

This script provides a forward-stepwise algorithm used to identify the best combination of features that maximizes the Fisher's distance between  two classes of data. Selected features  are synthetizied into a composite index (Cindex). 
In the example, the feratures are a a set of climate-related variables and the resulting composite index represent the combination of specific, selected features, that maximize the distance between exceptional high and low olive yields over Italy in the period 2006-2020.

Rationale and further details are presented in:
Di Paola, A., Di Giuseppe, E., Gutierrez, A P., Ponti, L., & Pasqui, M. (2023). Climate stressors modulate interannual olive yield at province level in Italy: A composite index approach to support crop management. Journal of Agronomy and Crop Science, 00, 1– 14. https://doi.org/10.1111/jac.12636

Please cite the aforementioned article  to fully exploiuy ther Creative Commons Licence

REQUIREMENTS:
- Python 3
- NumPy
- Matplotlib

HISTORY
v1. First release December 2021 (deprecated)

v2. Updated in October 2022
    - graphical bug fixed (blue box == data under neg_flag)
    - variable selection bug fixed 
    - added new features:
        > A new library utils.py.py has been created;
        > the recursive calculation for C-index has been put into the function Cindex within utils.py

v3. Updated in March 2025
    - corrected grammatical errors in comments
    - improved comments explaining some command lines
    - renamed some variables to improve the readability and comprehensibility of the code
