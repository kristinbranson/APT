function [P_all,w_all] = DeleteSample3DPointsRomain(r,c,S,w_in,cRig,K)
                        
  nviews = numel(r);
  cNames = {'L','R','B'}; % for now.
  P_all = nan(3,numel(r{1})+1,numel(r{2})+1,numel(r{3})+1,3);
  w_all = zeros(numel(r{1})+1,numel(r{2})+1,numel(r{3})+1,3);
  for view = 1:numel(r),
    fview = view;
    sview = mod(view,nviews)+1;
    oview = mod(view+1,nviews)+1;
    rfront = r{fview};
    rside = r{sview};
    rother = r{oview};
    cfront = c{fview};
    cside = c{sview};
    cother = c{oview};
    
    nfront = numel(cfront);
    nside = numel(cside);
    nother = numel(cother);
    w = zeros(nfront+1,nside+1,nother+1);
    pfront = nan(nfront,nside);
    pside = nan(nfront,nside);
    pother = nan(nfront,nside);
    P = nan(3,nfront+1,nside+1,nother+1);
    Sfront = S{fview};
    Sside  = S{sview};
    Sother  = S{oview};
    wfront = w_in{fview};
    wside  = w_in{sview};
    wother = w_in{oview};
    for ifront = 1:nfront,
      for iside = 1:nside,
%         [P(:,ifront,iside),~,~,~,xfront_re,xside_re] = dlt_2D_to_3D_point(dlt_front,dlt_side,xfront(:,ifront),xside(:,iside),...
%           'Sfront',Sfront(:,:,ifront),'Sside',Sside(:,:,iside));
%         pfront(ifront,iside) = mvnpdf(xfront_re,xfront(:,ifront)',Sfront(:,:,ifront));
%         pside(ifront,iside) = mvnpdf(xside_re,xside(:,iside)',Sside(:,:,iside));
%         w(ifront,iside) = pfront(ifront,iside)*pside(ifront,iside)*wfront(ifront)*wside(iside);

        cropfront = cRig.y2x([rfront(ifront) cfront(ifront)],cNames{fview});
        cropside = cRig.y2x([rside(iside) cside(iside)],cNames{sview});
        [P3d_front,P3d_side] = cRig.stereoTriangulate(cropfront,cropside,cNames{fview},cNames{sview});
        [rfront_re,cfront_re] = cRig.projectCPR(P3d_front,fview);
        [rside_re,cside_re] = cRig.projectCPR(P3d_side,sview);
        [rother_re,cother_re] = cRig.projectCPR(cRig.camxform(P3d_front,[cNames{fview} cNames{oview}]),oview);
        
        pfront(ifront,iside) = mvnpdf([cfront_re rfront_re],[cfront(ifront) rfront(ifront)],Sfront(:,:,ifront));
        pside(ifront,iside) = mvnpdf([cside_re rside_re],[cside(iside) rside(iside)],Sside(:,:,iside));
        
        oprobs = zeros(1,numel(cother)+1);
        for iother = 1:numel(cother)
          oprobs(iother) = mvnpdf([cother_re rother_re],[rother(iother) cother(iother)],Sother(:,:,iother));
        end
        oprobs(end) = 0.000001;  % in case the point isn't visible in the "other" view.
        [pother(ifront,iside),idxother] = max(oprobs);
        w(ifront,iside,idxother) = pfront(ifront,iside)*pside(ifront,iside)*wfront(ifront)*wside(iside)*pother(ifront,iside);
        curP = cRig.camxform(P3d_front,[cNames{fview} cNames{1}]);
        P(:,ifront,iside,idxother) = curP;
      end
    end
    for ixx = 1:(view-1)
      P = permute(P,[1 4 2 3]);
      w = permute(w,[3 1 2]);
    end
    P_all(:,:,:,:,view) = P;
    w_all(:,:,:,view) = w;
  end
  
  w_all = w_all / nansum(w_all(:));
  [~,order] = sort(w_all(:),1,'descend');

  P_all = P_all(:,order(1:K));
  w_all = w_all(order(1:K));
