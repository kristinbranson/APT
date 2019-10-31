function grayim=rgb2gray_cond(im)
if size(im,3)==3
    grayim=rgb2gray(im);
else
    grayim=im;
end