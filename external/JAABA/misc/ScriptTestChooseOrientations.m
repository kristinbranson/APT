%% parameters

x0 = 0;
y0 = 0;
theta0 = 0;
T = 1000;
dampen_pos = .25;
dampen_theta = .5;
sigma_pos = 10;
sigma_theta = .05;

% these are the parameters I use for tracking flies
velocity_angle_weight = .03;
max_velocity_angle_weight = .18;

%% initialize

x = nan(1,T);
y = nan(1,T);
theta = nan(1,T);

x(1) = x0;
y(1) = y0;
theta(1) = theta0;
dx = 0;
dy = 0;
dtheta = 0;

%% random walk

for t = 2:T,
  
  x(t) = x(t-1) + dx*(1-dampen_pos) + randn(1)*sigma_pos;
  y(t) = y(t-1) + dy*(1-dampen_pos) + randn(1)*sigma_pos;
  theta(t) = theta(t-1) + dtheta*(1-dampen_theta) + randn(1)*sigma_theta;
  speed = sqrt( (x(t)-x(t-1))^2 + (y(t)-y(t-1))^2 );
  dx = speed*cos(theta(t));
  dy = speed*sin(theta(t));
  dtheta = theta(t)-theta(t-1);
  theta(t) = modrange(theta(t),-pi,pi);
  
end

%% plot

figure(1);
clf;
ax = [min(x),max(x),min(y),max(y)];
ax = ax + [[-.01,.01]*(ax(2)-ax(1)),[-.01,.01]*(ax(4)-ax(3))];
l = max(ax(2)-ax(1),ax(4)-ax(3))*.025;

htrx = plot(x(1),y(1),'k.-');
hold on;
ho = plot(x(1),y(1),'ro','markerfacecolor','r');
hdir = plot(x(1)+[0,cos(theta(1))*l],y(1)+[0,sin(theta(1))*l],'m-');
axis equal;
axis(ax);
drawnow;

for t = 2:T,
  set(ho,'XData',x(t),'YData',y(t));
  set(hdir,'XData',x(t)+[0,cos(theta(t))*l],'YData',y(t)+[0,sin(theta(t))*l]);
  set(htrx,'XData',x(max(t-60,1):t),'YData',y(max(t-60,1):t));
  pause(.033);
end  

%% lose direction

theta_obs = modrange(theta,0,pi);

theta_reconstruct = choose_orientations(x,y,theta_obs,velocity_angle_weight,max_velocity_angle_weight);

fprintf('Max reconstruction error: %f\n',max(abs(theta-theta_reconstruct)));

%% plot

figure(1);
clf;
tmp = plot(repmat(1:T,[2,1]),[theta_reconstruct;theta],'.-');
xlabel('Time');
ylabel('Orientation vs reconstructed orientation');

%% plot

figure(1);
clf;
ax = [min(x),max(x),min(y),max(y)];
ax = ax + [[-.01,.01]*(ax(2)-ax(1)),[-.01,.01]*(ax(4)-ax(3))];
l = max(ax(2)-ax(1),ax(4)-ax(3))*.025;

htrx = plot(x(1),y(1),'k.-');
hold on;
ho = plot(x(1),y(1),'ro','markerfacecolor','r');
hdir = plot(x(1)+[0,cos(theta(1))*l],y(1)+[0,sin(theta(1))*l],'m-');
hdir_reconstruct = plot(x(1)+[0,cos(theta_reconstruct(1))*l],y(1)+[0,sin(theta_reconstruct(1))*l],'c--');
drawnow;
axis equal;
axis(ax);

for t = 2:T,
  set(ho,'XData',x(t),'YData',y(t));
  set(hdir,'XData',x(t)+[0,cos(theta(t))*l],'YData',y(t)+[0,sin(theta(t))*l]);
  set(hdir_reconstruct,'XData',x(t)+[0,cos(theta_reconstruct(t))*l],'YData',y(t)+[0,sin(theta_reconstruct(t))*l]);
  set(htrx,'XData',x(max(t-60,1):t),'YData',y(max(t-60,1):t));
  pause(.033);
end  