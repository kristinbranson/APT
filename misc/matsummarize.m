function t = matsummarize(fname)
s = whos('-file',fname);
s = rmfield(s,{'global' 'sparse' 'complex' 'nesting' 'persistent'});
t = struct2table(s);
t = t(:,{'name' 'bytes' 'size' 'class'});
t = sortrows(t,'bytes','descend');
