function SymbolicCopyStephenData(srcdir,destdir,varargin)

[regexpexclude] = myparse(varargin,'regexpexclude',{});

[~,n] = myfileparts(srcdir);
destdir1 = fullfile(destdir,n);

if ~exist(destdir1,'dir'),
  mkdir(destdir1);
end

[fs,infos] = mydir(srcdir);
for i = 1:numel(fs),
  if ~infos(i).isdir,
    [~,n] = myfileparts(fs{i});
    doexclude = false;
    for j = 1:numel(regexpexclude),
      doexclude = ~isempty(regexp(n,regexpexclude{j},'once'));
      if doexclude,
        break;
      end
    end
    if ~doexclude,
      unix(sprintf('ln -s %s %s',fs{i},fullfile(destdir1,n)));
    end
  else
    SymbolicCopyStephenData(fs{i},destdir1,varargin{:});
  end
end


