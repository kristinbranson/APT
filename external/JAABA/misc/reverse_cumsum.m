function c = reverse_cumsum(v)

c = reverse_vector(cumsum(reverse_vector(v)));