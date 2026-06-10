function s = structLeavesStr2Double(s, flds)
% Convert nonempty char-array leaves of struct s, named in flds, to doubles.
% Recurses into nested structs.
% flds: cellstr of fieldnames
for f = flds(:)', f = f{1};  %#ok<FXSET>
  val = s.(f) ;
  if isstruct(val)
    s.(f) = structLeavesStr2Double(s.(f), fieldnames(s.(f))) ;
  elseif ~isempty(val)
    if ischar(val)
      s.(f) = str2double(val) ;
    end
  else
    % none, empty
  end
end
end  % function
