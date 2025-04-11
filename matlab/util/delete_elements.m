function y = delete_elements(x, doomed_indices)
  % Delete elements from 1d array x.  This function hides the in-place mutation,
  % making it pure functional from the outside.
  y = x ;
  y(doomed_indices) = [] ; 
end
