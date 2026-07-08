# Project Name

Census Income (>50K) Classifier

# Business Objective

Predict whether an individual earns more than $50,000 per year from U.S. Census
demographic and employment attributes. The model supports fairness auditing,
targeted marketing eligibility, and socio-economic segmentation.

# Target Variable

`class` — 1 when annual income is greater than $50K, 0 otherwise.

# Key Drivers

Education level, occupation, hours worked per week, marital status, capital gains,
and age are typically the most influential factors in the prediction.

# Notes

Categorical features (workclass, education, occupation, etc.) are ordinal-encoded to
integer codes; the mapping from code to original category is documented in the data
dictionary under "Allowed Range".
