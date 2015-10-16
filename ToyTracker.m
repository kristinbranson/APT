classdef ToyTracker < handle
  
  methods 
    
    function trx = track(obj,trx,lpos,iPt,frm0,frm1) %#ok<INUSD,INUSL>
    % trx (input): current/existing trx (ignored for this tracker)
    % lpos: current labels (labeledpos): npts x 2 x nFrm x nTgt (expect
    %   labels only for first target)
    % iPt: desired pt(s) for (re)tracking (ignored here)
    % frm0: desired start frame for (re)tracking (ignored here)
    % frm1: desired end frame for (re)tracking (ignored here)
    %
    % trx (output): trajectories
    %
    % For this tracker, each label-point corresponds to one trx.
    
      [npts,ncoords,nfrms,ntgts] = size(lpos); %#ok<ASGLU>
      assert(ncoords==2);
      if ntgts~=1
        warningNoTrace('ToyTracker:ntgts',...
          'Using only labels for first target (npts=%d) to track.',npts);
      end
      
      % Toy Tracker always fully recreates trx
      trx = TrxUtil.createSimpleTrx(npts); 
      
      for iP = 1:npts
        % For each target, i) lock trajs to labeled frames, then ii)
        % interpolate
        
        lbls = squeeze(lpos(iP,:,:,1))'; % should be nfrm x 2
        tfLbledFrm = all(~isnan(lbls) & ~isinf(lbls),2);
        iLbledFrm = find(tfLbledFrm);
        
        if ~isempty(iLbledFrm)
          trx(iP) = TrxUtil.initStationary(trx(iP),...
            nan,nan,nan,iLbledFrm(1),iLbledFrm(end));
        end
        
        for i = 1:numel(iLbledFrm)-1
          % set traj in interval [iLbledFrm(i),iLbledFrm(i+1)]
          f0 = iLbledFrm(i);
          f1 = iLbledFrm(i+1);
          xy0 = lbls(f0,:);
          xy1 = lbls(f1,:);
          nfrmseg = f1-f0+1;
          trx(iP).x(f0:f1) = linspace(xy0(1),xy1(1),nfrmseg);
          trx(iP).y(f0:f1) = linspace(xy0(2),xy1(2),nfrmseg);
          trx(iP).theta(f0:f1) = nan(1,nfrmseg);
        end
      end
    end
    
  end
  
end