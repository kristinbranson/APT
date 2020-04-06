function h = polarerrorpatch(theta,r,r_err,color,varargin)

[isclosed,leftovers] = myparse_nocheck(varargin,'isclosed',true);

if isclosed,
  r = [r,r(1)];
  r_err = [r_err,r_err(1)];
  theta = [theta,theta(1)];
end

% bounds
plus_x = max(0,(r+r_err)).*cos(theta);
plus_y = max(0,(r+r_err)).*sin(theta);
minus_x = max(0,(r-r_err)).*cos(theta);
minus_y = max(0,(r-r_err)).*sin(theta);

% edge of patch
err_x = [plus_x,fliplr(minus_x)];
err_y = [plus_y,fliplr(minus_y)];

h = patch(err_x,err_y,color,leftovers{:});