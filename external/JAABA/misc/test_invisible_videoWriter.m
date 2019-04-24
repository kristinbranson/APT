addpath(genpath('/home/kristin/software/videoIO'));

%vw = avifile('test.avi');
vw = videoWriter('writertest.avi', ...
                     'width',320, 'height',240, 'codec','xvid');
hfig = 1;
figure(hfig);
clf;
set(1,'visible','off');
img = imread('peppers.png');
[nr,nc,three] = size(img);
h = fspecial('motion',10,5);
theta = linspace(0,2*pi,100);
r = 100;
for i=1:100
  set(0,'CurrentFigure',1);
  if i == 1,
    himg = imagesc(img); axis image; axis off;
    haxes = gca;
    hold on;
    hplot = plot([nc/2,nc/2+r*cos(theta(i))],[nr/2,nr/2+r*sin(theta(i))],'b','linewidth',4);
  else
    img = imfilter(img, h);
    set(himg,'cdata',img);
    set(hplot,'xdata',[nc/2,nc/2+r*cos(theta(i))],'ydata',[nr/2,nr/2+r*sin(theta(i))]);
  end
  kaddframe(vw,haxes);
end
vw=close(vw);
