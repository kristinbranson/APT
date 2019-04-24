function info = GetMATLABClusterInfo()

[~,me] = unix('whoami');
me = strtrim(me);
info.TMP_ROOT_DIR = fullfile('/scratch',me);

tmp = ver('MATLAB');
v = ['v',strrep(tmp.Version,'.','')];
info.MCR = fullfile('/groups/branson/bransonlab/share/MCR',v);
