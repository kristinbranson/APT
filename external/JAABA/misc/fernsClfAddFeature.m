function varargout = fernsClfAddFeature( data, hs, ferns, new_features, old_features)

[N,F]=size(data); assert(length(hs)==N);
M=size(ferns.inds,2); S=log2(size(ferns.pFern,1));
assert(N==size(ferns.inds,1));
assert(all(new_features<=F));
assert(all(old_features<=F));
nnew = numel(new_features);
nold = numel(old_features);

% replace with probability Preplace
Preplace = nnew/(nnew+nold);
doreplace = rand(M,S)<=Preplace;

s = struct;
s.doreplace = doreplace;
s.features_replace = new_features;

varargout = cell(1,nargout);
[varargout{:}] = fernsClfReplaceFeature(data,hs,ferns,[],s);