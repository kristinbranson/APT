function I = bwrotate(I,theta,iscleanup)

% get out of degrees
theta = theta*pi/180;

% find all 1's in the image
[y,x] = find(I);
x = x(:); y = y(:);
n = length(y);

% rotation matrix
R = [cos(theta),  sin(theta),
     -sin(theta), cos(theta)];
   
% center of rotation
[nr,nc] = size(I);
mux = nc/2;
muy = nr/2;
   
% rotate 1's positions
p = R * [x-mux,y-muy]' + repmat([mux;muy],[1,n]);

% round 1's positions
r = round(p(2,:));
c = round(p(1,:));

% create new image
r1 = min(r);
r2 = max(r);
c1 = min(c);
c2 = max(c);
offsetr = r1 - 1;
offsetc = c1 - 1;
nr = r2 - offsetr;
nc = c2 - offsetc;
I = false(nr,nc);
I(sub2ind([nr,nc],r-offsetr,c-offsetc)) = true;

% clean up?
if exist('iscleanup') & iscleanup,
  I = bwmorph(imclose(I,ones(3)),'skel',inf);
end;