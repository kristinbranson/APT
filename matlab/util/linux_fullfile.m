function result = linux_fullfile(varargin)
% Like fullfile(), but returns linux-style paths regardless of whether
% input paths use forward slash or backslash.  This will leave a
% Windows-style leading drive letter intact.

% Insert / between the elements of varargin, making a single string
result = '' ;
if nargin==0 ,
  return
end
% remove empty arguments
idx = ~cellfun(@isempty,varargin);
if ~any(idx),
  return;
end
args = varargin(idx);
protoresult_so_far = args{1} ;
for i = 2 : numel(args) ,
  protoresult_so_far = horzcat(protoresult_so_far, '/', args{i}) ;  %#ok<AGROW>
end

% Translate \s to /s
protoresult_2 = regexprep(protoresult_so_far, '\', '/') ;

% Replace repeated /s with a single /
result = remove_repeated_slashes(protoresult_2) ;

end
