function trx = ComputeRelativeLandmarkPos_Body(trx)

pos2 = trx.pos;
nfrm = numel(trx.bodytrx.x);
t0 = trx.bodytrx.firstframe; t1 = trx.bodytrx.endframe;
pos2(:,1,t0:t1) = pos2(:,1,t0:t1) - reshape(trx.bodytrx.x,[1,1,nfrm]);
pos2(:,2,t0:t1) = pos2(:,2,t0:t1) - reshape(trx.bodytrx.y,[1,1,nfrm]);
c = reshape(cos(trx.bodytrx.theta-pi/2),[1,1,nfrm]);
s = reshape(sin(trx.bodytrx.theta-pi/2),[1,1,nfrm]);
trx.pos_body = nan(size(trx.pos));
trx.pos_body(:,1,t0:t1) = pos2(:,1,t0:t1).*c+pos2(:,2,t0:t1).*s;
trx.pos_body(:,2,t0:t1) = -pos2(:,1,t0:t1).*s+pos2(:,2,t0:t1).*c;

% current frame in previous frame's body coord system
if nfrm == 0,
  trx.pos_body_prev = trx.pos_body;
else
  pos2 = trx.pos;
  pos2(:,1,t0+1:t1) = pos2(:,1,t0+1:t1) - reshape(trx.bodytrx.x(1:end-1),[1,1,nfrm-1]);
  pos2(:,2,t0+1:t1) = pos2(:,2,t0+1:t1) - reshape(trx.bodytrx.y(1:end-1),[1,1,nfrm-1]);
  trx.pos_body_prev = nan(size(trx.pos));
  trx.pos_body_prev(:,1,t0+1:t1) = pos2(:,1,t0+1:t1).*c(1:end-1)+pos2(:,2,t0+1:t1).*s(1:end-1);
  trx.pos_body_prev(:,2,t0+1:t1) = -pos2(:,1,t0+1:t1).*s(1:end-1)+pos2(:,2,t0+1:t1).*c(1:end-1);
end

