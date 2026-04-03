function result = isShortDLNetTypesRowArray(netTypes)
% Test whether netTypes is a DLNetType row array with 1 or 2 elements.
% This is what most the the DlNetTypes arrays in APT are.

n = numel(netTypes) ;
result = isa(netTypes, 'DLNetType') && isrow(netTypes) && 1<=n && n<=2 ;

end
