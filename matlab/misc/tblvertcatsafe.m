function t = tblvertcatsafe(t1,t2)
% Vertical concatenation of possibly dissimilar tables 
%
% Like regular vertcat, except any new field(s) in t1 relative to t2 or 
% vice versa are added as necessary with 'null' values.

f1 = tblflds(t1);
f2 = tblflds(t2);
fnew = setxor(f1,f2);
for f=fnew(:)',f=f{1}; %#ok<FXSET>
  if ~any(strcmp(f1,f))
    v = t2.(f);
    if isnumeric(v)
      width = size(v,2);
      t1.(f) = nan(height(t1),width);
    else
      error('Unsupported table value for field ''%s''.',f);
    end
  else
    v = t1.(f);
    if isnumeric(v)
      width = size(v,2);
      t2.(f) = nan(height(t2),width);
    else
      error('Unsupported table value for field ''%s''.',f);
    end    
  end
end
  
t = [t1;t2];
