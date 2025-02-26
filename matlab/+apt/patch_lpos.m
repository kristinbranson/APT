function result = patch_lpos(lpos)
% if any points in lpos are nan, set them to be somewhere.

% convhull() throws sometimes, so we wrap in a try-catch to just ignore that.
try
  ismissing = any(isnan(lpos),2);
  if nnz(~ismissing) > 1 && any(ismissing) ,
    k = convhull(lpos(~ismissing,1),lpos(~ismissing,2));
    result = lpos ;
    for j = find(ismissing)',
      i1 = randsample(numel(k)-1,1);
      i2 = i1 + 1;
      lambda = .25+.5*rand(1);
      p = result(k(i1),:)*lambda + result(k(i2),:)*(1-lambda);
      result(j,:) = p;
    end
  else
    result = lpos ;
  end
catch me
  if strcmp(me.identifier, 'MATLAB:convhull:NonFiniteInputPtsErrId')
    %warning('APT:convhull:NonFiniteInputPtsErrId', '%s', me.message) ;
    result = lpos ;  % give up and return the original
  else
    rethrow(me) ;
  end
end

end  % function
