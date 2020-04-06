function c = numarr2trimcellstr(x)
c = strtrim(cellstr(num2str(x(:))));
