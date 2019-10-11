function [dupcats,idupcats] = finddups(x,varargin)
% Look for duplicates in rows of x
%
% x: [Nxncol] numerical or cellstr
%
% dupcats: [Ndupsetsx1] cell. dupcats{i} contains identified/matching 
%   rows of x for duplicate category i
% idupcats: [Nx1] indices into dupcats. idupcats(i) is:
%   * nan if x(i,:) is not a dup
%   * an index into dupcats if x(i,:) is a dup
%
% numeric x's are compraed with ismember(...'rows'); NaNs are NOT counted
% as a match. 

verbose = myparse(varargin,...
  'verbose',false);

N = size(x,1);
dupcats = cell(0,1);
idupcats = nan(N,1);

if isnumeric(x)
  eqfcn = @isequaln;
elseif iscellstr(x)
  eqfcn = @isequal;
else
  error('Unsupported type.');
end

for i=1:N
  if verbose && mod(i,50)==0
    disp(i);
  end
    
  if ~isnan(idupcats(i))
    % this row has already been marked as dup
    continue;
  end
  tf = arrayfun(@(j)eqfcn(x(i,:),x(j,:)),1:N);
  irows = find(tf);
  switch numel(irows)
    case 0
      assert(false,'Row found that doesn''t match itself.');
    case 1
      assert(eqfcn(x(i,:),x(i,:))); % confirm that row matches itself as expected
    otherwise % duplicates found
      dupcats{end+1,1} = irows; %#ok<AGROW>
      idupcats(irows) = numel(dupcats);
  end
end
