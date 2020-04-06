function h = polarimagesc(edges_r,edges_theta,center,v,varargin)

ninterptheta = 5;
nbinsr = length(edges_r)-1;
nbinstheta = length(edges_theta)-1;

xplot = nan(2*ninterptheta,nbinsr,nbinstheta);
yplot = nan(2*ninterptheta,nbinsr,nbinstheta);
for binr = 1:nbinsr,
  for bintheta = 1:nbinstheta,
    rplot = [repmat(edges_r(binr+1),[1,ninterptheta]),repmat(edges_r(binr),[1,ninterptheta])];
    thetainterp = linspace(edges_theta(bintheta),edges_theta(bintheta+1),ninterptheta);
    thetaplot = [thetainterp,fliplr(thetainterp)];
    xplot(:,binr,bintheta) = rplot.*cos(thetaplot);
    yplot(:,binr,bintheta) = rplot.*sin(thetaplot);
  end
end
xplot = xplot + center(1);
yplot = yplot + center(2);
faces = reshape(1:nbinsr*nbinstheta*2*ninterptheta,[2*ninterptheta,nbinsr*nbinstheta])';
verts = [xplot(:),yplot(:)];
h = patch('Faces',faces,'Vertices',verts,'FaceColor','flat','FaceVertexCData',v(:),'EdgeColor','none',varargin{:});