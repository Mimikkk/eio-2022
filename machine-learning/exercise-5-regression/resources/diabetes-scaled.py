def linear_regression(x, y):
  from numpy import mean

  a = (mean(x) * mean(y) - mean(x * y)) / (mean(x) ** 2 - mean(x ** 2))
  b = mean(y) - a * mean(x)
  return a, b
