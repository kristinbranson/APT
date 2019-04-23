function [imcrop,A] = CropImAroundTrx(im,x,y,theta,winwidth,winheight,varargin)

[interp,fillvalues] = myparse(varargin,'interp','bilinear','fillvalues',0);

imsz = size(im);

T = [1,0,0
  0,1,0
  -x,-y,1];

R = [cos(theta+pi/2),-sin(theta+pi/2),0
  sin(theta+pi/2),cos(theta+pi/2),0
  0,0,1];

A = T*R;
tform = maketform('affine',A);

imcrop = imtransform(im,tform,interp,'udata',[1,imsz(2)],'vdata',[1,imsz(1)],...
  'xdata',[-winwidth,winwidth],'ydata',[-winheight,winheight],'fillvalues',fillvalues);