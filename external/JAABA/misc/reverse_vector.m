function v = reverse_vector(v)

s = size(v);
v = reshape(flipud(v(:)),s);