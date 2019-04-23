function v = isnaturalnumber(x)

v = isnumeric(x) & x > 0 & round(x) == x;