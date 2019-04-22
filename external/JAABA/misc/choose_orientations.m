% theta = choose_orientations(x,y,phi,velocity_angle_weight,[max_velocity_angle_weight=1])
%
% we will set the orientation to theta_t = phi_t + s_t * pi
% we want to choose s_t to minimize
% \sum_t cost(s_t|s_{t-1})
% cost(s_t|s_{t-1}) = [(1 - w(||v_t||^2))*d(\theta_t,\theta_{t-1}) +
%                      w(||v_t||^2)*d(\theta_t,angle(v_t))]
% where w(||v_t||^2) = \min{max_velocity_angle_weight, velocity_angle_weight*||v_t||^2}
%
% we will find the most likely states s_t using the recursion
% cost_t(s_t) = min_{s_{t-1}} { cost_{t-1}(s_{t-1}) + cost(s_t|s_{t-1})
%
% Inputs:
% x: N x 1 vector where x(t) is the x-coordinate of the center of the fly
% at time t
% y: N x 1 vector where y(t) is the y-coordinate of the center of the fly
% at time t
% phi: N x 1 vector where phi(t) is the orientation of the fly at time t
% velocity_angle_weight: relative weight coefficient of the velocity term
% max_velocity_angle_weight (by default = 1): maximum relative weight of
% the velocity term. 
%
function theta = choose_orientations(x,y,theta,velocity_angle_weight,max_velocity_angle_weight)

if ~exist('max_velocity_angle_weight','var'),
  max_velocity_angle_weight = 1;
end

inputsz = size(x);
x = x(:);
y = y(:);
theta = theta(:);
max_velocity_angle_weight = max_velocity_angle_weight(:);

% number of frames
N = length(x);

% allocate space for storing the optimal path
stateprev = zeros(N-1,2);

% allocate space for computing costs
tmpcost = zeros(2,1);
costprevnew = zeros(2,1);

% initialize first frame
costprev = zeros(2,1);

% compute velocity
vx = [0;diff(x)];
vy = [0;diff(y)];
v = sqrt(vx.^2+vy.^2);

% compute angle of velocity
velocityangle = atan2(vy,vx);

% compute weight for velocity term
w = min(max_velocity_angle_weight,velocity_angle_weight.*v);

% compute iteratively
for t = 2:N,
  
  % compute for both possible states
  for scurr = 1:2,
    
    % try both previous states
    thetacurr = theta(t) + (scurr-1)*pi;
    
    for sprev = 1:2,
      
      thetaprev = theta(t-1) + (sprev-1)*pi;
      costcurr = (1-w(t))*angledist(thetaprev,thetacurr) + ...
        w(t)*angledist(thetacurr,velocityangle(t));
      tmpcost(sprev) = costprev(sprev) + costcurr;
      
    end
    
    % choose the minimum
    sprev = argmin(tmpcost);
    
    % set pointer for path
    stateprev(t-1,scurr) = sprev;
    
    % set cost
    costprevnew(scurr) = tmpcost(sprev);
    
  end
  
  % copy over
  costprev(:) = costprevnew(:);
  
end

% choose the best last state
scurr = argmin(costprev);

if scurr == 2,
  theta(end) = modrange(theta(end)+pi,-pi,pi);
end

% choose the best states
for t = N-1:-1:1,
  scurr = stateprev(t,scurr);
  if scurr == 2,
    theta(t) = modrange(theta(t)+pi,-pi,pi);
  end  
end

theta = reshape(theta,inputsz);