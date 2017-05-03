function mstr = StephenVideo2Identifier(moviename)
ss = strsplit_jaaba(moviename,'/');
mstr = [ss{end-5},'__',ss{end-2},'__',ss{end}(end-10:end-6)];
