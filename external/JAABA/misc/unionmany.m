% c = unionmany(set1,set2,...,setn,['rows'])
function c = unionmany(varargin)

if ischar(varargin{end}) && strcmpi(varargin{end},'rows'),
  rows = {'rows'};
  varargin = varargin(1:end-1);
else
  rows = {};
end

if length(varargin) < 2,
  error('Usage: c = unionmany(a1,a2,...,an) or c = unionmany(a1,a2,...,an,''rows'')');
end

c = varargin{1};
for i = 2:length(varargin),
  c = union(c,varargin{i},rows{:});
end

    