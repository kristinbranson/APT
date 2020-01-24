function m2 = mapKeyRemap(m,keyOrig2New)
% Remap keys of a containers.Map
%
% m: containers.Map
% keyOrig2New: containers.Map. keyOrig2New(oldKey)==newKey where oldKey and
%   newKey are of the type m.KeyType. keyOrig2New must contain all keys of 
%   m.
%
% m2: new containers.Map. For each key in m, knew=keyOrig2New(k) gives 
%   a new key. m2 has the same contents as m, except:
%   1. When knew~=0, the old key k is replaced by the new key k2.
%   2. When knew==0, the old key k and its value are removed/(not present 
%      in m2).

m2 = containers.Map('KeyType',m.KeyType,'ValueType',m.ValueType);
keysold = m.keys;
for i=1:numel(keysold)
  k = keysold{i};
  k2 = keyOrig2New(k);
  if k2==0
    % none
  else
    v = m(k);
    m2(k2) = v;
  end
end