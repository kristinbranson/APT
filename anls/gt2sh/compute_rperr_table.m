function [tfDLT,rpe] = compute_rperr_table(t,pfld,fly2calibMap)

tfFly2Calib = exist('fly2calibMap','var')>0;

n = height(t);
tfDLT = true(n,1);
rpe = nan(n,5,2);
for i=1:n
  if mod(i,50)==0
    disp(i);
  end

  pLbl = t.(pfld)(i,:);
  
  xyLbl = reshape(pLbl,[10 2]); %pLbl is all xs then all ys
  uvLbl = cat(3,xyLbl(1:5,:),xyLbl(6:10,:)); % [5x2x2]. pt,(x/y),vw
  uvLbl = permute(uvLbl,[2 1 3]); % [2x5x2]. (x,y), ipt, vw

  if ~tblfldscontains(t,'vcd')
    fly = t.flyID(i);
    calibFile = fly2calibMap(fly);
    try
      crObj0 = CalRig.loadCreateCalRigObjFromFile(calibFile);
    catch ME
      fprintf(2,'row %d: %s\n',i,ME.message);
      % tfDLT(i) will be true; rpe will be nan
      continue;
    end   
  else
    crObj0 = t.vcd{i};
    if isempty(crObj0)
      % try fly
      fly = t.flyID(i);
      try
        calibFile = fly2calibMap(fly);
        crObj0 = CalRig.loadCreateCalRigObjFromFile(calibFile);
      catch ME
        fprintf(2,'row %d fly %d: %s\n',i,fly,ME.message);
        % tfDLT(i) will be true; rpe will be nan
        continue;
      end
    end
  end
  if isa(crObj0,'CalRigSH') 
    % none
  elseif isa(crObj0,'OrthoCamCalPair')
    tfDLT(i) = false;
  else
    assert(false);
  end
  
  if tfDLT(i)
    [~,~,rpetmp] = crObj0.triangulate(uvLbl);
    rpe(i,:,:) = rpetmp;
  else    
    [~,~,~,~,rpeL,rpeR] = crObj0.stereoTriangulate(...
      uvLbl(:,:,1),uvLbl(:,:,2));
    rpe(i,:,1) = rpeL;
    rpe(i,:,2) = rpeR;
  end
end