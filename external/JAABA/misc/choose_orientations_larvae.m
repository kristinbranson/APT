% doflip = choose_orientations(spinex,spiney,[movement_weight=.5,[min_segment_length=5]])
%
% chooses whether to flip or not flip the spine in order to minimize:
% \sum_t cost(s_t|s_{t-1})
% cost(s_t|s_{t-1}) = movement_weight*||spine_t(s_t)-spine_{t-1}(s_{t-1})|| +
%                     (1-movement_weight)*< segmentdir_{t-1}(s_{t-1}), 
%                                           spine_t(s_t)-spine_{t-1}(s_{t-1}) >
%
% we will find the most likely states s_t using the recursion
% cost_t(s_t) = min_{s_{t-1}} { cost_{t-1}(s_{t-1}) + cost(s_t|s_{t-1})
% If any segment is less than length min_segment_length, its direction cost
% is not included, and the weights are adjusted to reflect this. 
%
% Inputs:
% spinex: nspinepts x N vector where spinex(:,t) is the x-coordinates of the
% spine of the larva, and spinex(1,t) should be the x-coordinate of the
% head of the larva at time t.
% spiney: nspinepts x N vector where spiney(:,t) is the y-coordinates of the
% spine of the larva, and spiney(1,t) should be the y-coordinate of the
% head of the larva at time t.
% movement_weight: relative weight of the movement portion of the
% criterion. default = .5.
% min_segment_length: minimum segment length required to use the segment
% direction. default = 5
%
% Example:
%   spinepts = [1,6,11];
%   doflip = choose_orientations(trx(i).xspine(spinepts,:),trx(i).yspine(spinepts,:),.5,10);


function doflip = choose_orientations_larvae(spinex,spiney,movement_weight,min_segment_length)

if nargin < 3,
  movement_weight = .5;
end
if nargin < 4,
  min_segment_length = 5;
end

direction_weight = 1 - movement_weight;

% number of points in the spine, number of frames
[nspinepts,N] = size(spinex);

% allocate space for storing the optimal path
stateprev = zeros(N-1,2);

% allocate space for computing costs
tmpcost = zeros(2,1);
costprevnew = zeros(2,1);

% initialize first frame
costprev = zeros(2,1);

% precompute stuff

% pre-flip
spinex_flip = flipud(spinex);
spiney_flip = flipud(spiney);

% direction
dx = diff(spinex,1,1);
dy = diff(spiney,1,1);
dx(end+1,:) = dx(end,:);
dy(end+1,:) = dy(end,:);
z = sqrt(dx.^2 + dy.^2);
dx = dx ./ z;
dy = dy ./ z;
islongenough = z >= min_segment_length;


dx_flip = diff(spinex_flip,1,1);
dy_flip = diff(spiney_flip,1,1);
dx_flip(end+1,:) = dx_flip(end,:);
dy_flip(end+1,:) = dy_flip(end,:);
z = sqrt(dx_flip.^2 + dy_flip.^2);
dx_flip = dx_flip ./ z;
dy_flip = dy_flip ./ z;
islongenough_flip = z >= min_segment_length;

% compute iteratively
for t = 2:N,
  
  % compute for both possible states
  for scurr = 1:2,

    if scurr == 1,
      xcurr = spinex(:,t);
      ycurr = spiney(:,t);
    else
      xcurr = spinex_flip(:,t);
      ycurr = spiney_flip(:,t);
    end
      
    
    % try both previous states    
    for sprev = 1:2,
      
      if sprev == 1,
        xprev = spinex(:,t-1);
        yprev = spiney(:,t-1);
        dxcurr = dx(:,t-1);
        dycurr = dy(:,t-1);
        islongenough_curr = islongenough(:,t-1);
      else
        xprev = spinex_flip(:,t-1);
        yprev = spiney_flip(:,t-1);
        dxcurr = dx_flip(:,t-1);
        dycurr = dy_flip(:,t-1);
        islongenough_curr = islongenough_flip(:,t-1);
      end
      
      vxcurr = xcurr - xprev;
      vycurr = ycurr - yprev;
      
%       dxcurr = diff(xprev,1,1);
%       dycurr = diff(yprev,1,1);
%       z = sqrt(dxcurr.^2 + dycurr.^2);
%       dxcurr = dxcurr ./ z;
%       dycurr = dycurr ./ z;
%       dxcurr(end+1) = dxcurr(end);
%       dycurr(end+1) = dycurr(end);
      
      movement_cost = sqrt(sum( vxcurr.^2 + vycurr.^2, 1));
      nlongenough_curr = nnz(islongenough_curr);
      if nlongenough_curr == 0,
        costcurr = movement_cost;
      else
        direction_cost = sum( dxcurr(islongenough_curr).*vxcurr(islongenough_curr) + ...
          dycurr(islongenough_curr).*vycurr(islongenough_curr) );
        % since we are only using part of the direction cost, the total
        % weight will be smaller
        costcurr = (movement_cost*movement_weight + direction_cost*direction_weight) / ...
          (movement_weight + direction_weight*(nlongenough_curr/nspinepts));
      end

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

doflip = false(1,N);
doflip(end) = scurr == 2;

% choose the best states
for t = N-1:-1:1,
  scurr = stateprev(t,scurr);
  doflip(t) = scurr == 2;
end

