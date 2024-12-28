# Box-Cox Transformation

The **Box-Cox transformation** is a mathematical technique used to stabilize variance and make data more normally
distributed. It is particularly useful in situations where:

- Your data is heavily skewed.
- Variance is not constant across the range of the data (heteroscedasticity).

### Formula

The transformation depends on a parameter - λ (lambda):
$$y(\lambda) =
\begin{cases}
\frac{y^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \\
\log(y) & \text{if } \lambda = 0
\end{cases}
$$

Where:

- y: Original data.
- y′: Transformed data.

### Key Points

- It works only with positive data. If your data includes zero or negative values, you need to add a constant to make
  all values positive before applying the transformation.
- The λ parameter is typically determined automatically to minimize the skewness of the transformed data.
- When $𝜆=0$ the transformation is equivalent to taking the natural logarithm.

### Uses

- Preparing data for statistical analysis or machine learning models.
- Stabilizing variance in time-series forecasting.
- Normalizing data distributions for hypothesis testing.

# Winsorization

**Winsorization** is a data preprocessing technique used to limit extreme values in a dataset by capping them to
specified percentiles. It does not remove outliers; instead, it replaces them with less extreme values.

### How It Works

For a given dataset, values above a certain upper percentile or below a certain lower percentile are replaced with the
nearest value within those percentiles:

- For example, if you Winsorize a dataset at the 5th and 95th percentiles:
    - Values below the 5th percentile are set to the value at the 5th percentile.
    - Values above the 95th percentile are set to the value at the 95th percentile.

**Example**

- Original data: $[1, 2, 3, 4, 5, 100]$
- Winsorized at 95th percentile: $[1, 2, 3, 4, 5, 5]$

### Uses

- Reducing the influence of extreme values on statistical metrics (e.g., mean or standard deviation).
- Preventing outliers from distorting results in regression analysis or machine learning models.

## Comparison

| Feature              | Box-Cox Transformation                   | Winsorization                           |
|----------------------|------------------------------------------|-----------------------------------------|
| Purpose              | Stabilize variance, normalize data.      | 	Limit the influence of extreme values. |
| Technique	           | Applies a mathematical transformation.	  | Caps values at certain percentiles.     |                                                              
| Effect on Data Shape | 	Changes the overall shape of the data.	 | Retains most of the original shape.     |                                                                       
| Use Case	            | When data needs normalization.	          | When outliers are problematic.          |

Both techniques are often used in preprocessing pipelines for improved statistical and modeling performance.