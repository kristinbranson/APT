function [tf,x] = getmatv(fname)
x = evalc(['type(''', fname, ''')']);
x = x(2:20);
tf = strcmp(x, 'MATLAB 7.3 MAT-file');
