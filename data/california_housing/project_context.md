# Project Name

California Median House Value Regressor

# Business Objective

Estimate the median house value of a California census block group from location,
housing, and demographic attributes. The model supports property valuation,
investment screening, and regional pricing analysis.

# Target Variable

Median house value for the block group, expressed in hundreds of thousands of US
dollars (e.g. 2.5 = $250,000).

# Key Drivers

Median income of the block group, geographic location (latitude/longitude), average
number of rooms per household, and house age are typically the strongest predictors.

# Notes

All eight features are numeric and continuous; no encoding is applied. Values are
aggregated at the census block-group level rather than per individual home.
