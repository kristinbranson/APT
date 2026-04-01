function y = replaceAt(x, indices, yAtIndices)
% Return x with elements at indices indices replaced by yAtIndices. This
% exists becasue sometimes it's nice to write things in a functional style.

y = x ;
y(indices) = yAtIndices ;

end  % function
