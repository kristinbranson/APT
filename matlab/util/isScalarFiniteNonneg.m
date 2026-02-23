function result = isScalarFiniteNonneg(v)
% Returns true iff v is a scalar, finite, nonnegative number.

result = isscalar(v) && isnumeric(v) && isfinite(v) && v >= 0;
