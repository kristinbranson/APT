function rgb2 = rgbbrighten(rgb,f)
% rgb/rgb2: [Nx3]
% f: scalar in [0,1]
rgb2 = rgb + f*(1-rgb);
