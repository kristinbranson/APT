function [p,node,cname] = treeprob(tree,x,varargin)

if nargout > 2,
  [yfit,node,cname] = treeval(tree,x,varargin{:});
else,
  [yfit,node] = treeval(tree,x,varargin{:});
end;
[nr,nc] = size(yfit);
p = tree.classprob(node(:),:);
p = permute(reshape(p,[nr,nc,tree.nclasses]),[1,3,2]);
