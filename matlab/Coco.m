classdef Coco < handle
  properties
    j
    fname
  end
  methods
    function obj = Coco(jsonfname)
      if exist(jsonfname,'file')==0
        jsonfname = fullfile(pwd,jsonfname);
      end
      if exist(jsonfname,'file')==0
        error('Cannot find file: %s',jsonfname);
      end
      txt = readtxtfile(jsonfname);
      obj.j = jsondecode(txt{1});
      obj.fname = jsonfname;
    end
    function [imids,imidscnt] = summary(obj)
      ims = obj.j.images;
      nims = numel(ims);
      
      imids = [ims.id];
      assert(isequal(imids,unique(imids)));
      assert(numel(imids)==nims);
      
      tfcontiguous = isequal(imids,0:nims-1);
      fprintf('%d images, contiguous=%d.\n',nims,tfcontiguous);
      if ~tfcontiguous
        fprintf('first imid=%d; last imid=%d.\n',imids(1),imids(end));
      end
      
      anns = obj.j.annotations;
      nann = numel(anns);
      annimids = [anns.image_id];
      assert(nann==numel(annimids));
      imidscnt = arrayfun(@(x)nnz(annimids==x),imids);
      fprintf('%d annotations.\n',nann);
      fprintf('breakdown of imid counts in anns:\n');
      summary(categorical(imidscnt));
    end
    function save(obj,fname)
      if nargin==1 || isempty(fname)
        fname = obj.fname; %#ok<NASGU>
      end
%       nowstr = datestr(now,'yyyymmddTHHMMSS');
%       bakfname = sprintf('fname.%s',nowstr);
%       assert(exist(bakfname,'file')==0);
%       [st,msg] = movefile(fname,bakfnme);
%       if st==0
%         error(msg);
%       else        
%         fprintf('Backed up %s->%s.\n',fname,bakfname);
%       end
      txt = jsonencode(obj.j);
      assert(exist(fname,'file')==0);
      fh = fopen(fname,'w');
      fprintf(fh,'%s\n',txt);
      fclose(fh);
      fprintf('Wrote %s.\n',fname); 
    end
    function computeBBoxAndAreaFromKps(obj,varargin)
      % sets .bbox, .area from .keypoints.
      
      [type,computeRoiArgs,idxanns] = myparse(varargin,...
        'type','simple',... % 'simple' or 'apt'
        'computeRoiArgs',{}, ... % used if style=='apt'
        'idxanns',[] ...
        );
      
      anns = obj.j.annotations;
      nann = numel(anns);
      if isempty(idxanns)
        idxanns = 1:nann;
      end
      for i=idxanns(:)'
        kps = reshape(anns(i).keypoints,3,[])';
        %assert(all(kps(:,3)==2));
        kps = kps(:,1:2); % npt x 2
        switch type
          case 'simple'
            minxy = min(kps);
            maxxy = max(kps);
            dxy = maxxy-minxy;
            bbox = [minxy dxy];            
          case 'apt'            
            roi = Coco.computeRoi(kps,computeRoiArgs{:});
            minxy = min(roi);
            maxxy = max(roi);
            dxy = maxxy-minxy;
            bbox = [minxy dxy];
        end
        anns(i).bbox = bbox(:);
        anns(i).area = bbox(3)*bbox(4);
      end
      obj.j.annotations = anns;
      fprintf('Updated %d anns.\n',numel(idxanns));
    end
    function massageImageFilepaths(obj,pat,rep)
      nim = numel(obj.j.images);
      for i=1:nim
        obj.j.images(i).file_name = regexprep(obj.j.images(i).file_name,...
          pat,rep);
      end
      fprintf('Regexp''d %d filenames.\n',nim);      
    end
    function massage(obj)
      obj.j.annotations = rmfield(obj.j.annotations,'iscrowd');
      for i=1:numel(obj.j.annotations)
        obj.j.annotations(i).segmentation = [];
      end
      fprintf(2,'HAND MASSAGE CATEGORIES\n');
    end
    function rmExtra(obj)
      % rm extraneous flds that cause errors in PyCoco
      jj = obj.j;
      jj = rmfield(jj,'categories');
      jj.annotations = rmfield(jj.annotations,{'segmentation' 'keypoints'});
      obj.j = jj;
    end
    function viz(obj,imidx,varargin)

      [scargs,ttlargs,showkps,kpsplotspec,kpsmrkrsz,fignum] = myparse(varargin,...
        'scargs',{16}, ...
        'ttlargs',{'fontsize',16,'fontweight','bold','interpreter','none'}, ...
        'showkps',true, ...
        'kpsplotspec','g.', ...
        'kpsmrkrsz',14, ...
        'fignum',11 ...
        );
      
            
      hfig = figure(fignum);
      ims = obj.j.images;
      nim = numel(ims);
      anns = obj.j.annotations;
      
      if isempty(imidx)
        imidx = 1:nim;
      end
      for iim=imidx(:)'
        imid = ims(iim).id;
        imfname = ims(iim).file_name;
        im = imread(imfname);
        
        clf;
        ax = axes;
        imagesc(im);
        colormap gray;
        hold on;
        axis square;
        
        jann = find([anns.image_id]==imid);
        nannim = numel(jann);        
        for j = jann(:)'
          bb = anns(j).bbox;
          bb(3) = bb(1)+bb(3);
          bb(4) = bb(2)+bb(4);
          plot(bb([1 1 3 3 1]),bb([2 4 4 2 2]),'r-','linewidth',3);
          
          if showkps
            kps = anns(j).keypoints;
            kps = reshape(kps,3,[]).';
            plot(kps(:,1),kps(:,2),kpsplotspec,'markersize',kpsmrkrsz);
          end
        end
        
        tstr = sprintf('%d: imid %d. %d tgts',iim,imid,nannim);
        title(tstr,ttlargs{:});
        input(num2str(iim));
      end        
    end
    function [roi,kps] = getann(obj,ianns,varargin)
      % roi: [nann x 4]
      % kps: [npt x 2 x nann]
      
      kpsfull = myparse(varargin,...
        'kpsfull', false ...
        );
      
      if nargin==1 || isempty(ianns)
        ianns = 1:numel(obj.j.annotations);
      end
        
      nann = numel(ianns);
      roiall = nan(nann,4);
      npts = numel(obj.j.annotations(1).keypoints)/3;
      if kpsfull
        kpsall = nan(npts,3,nann);
      else
        kpsall = nan(npts,2,nann);
      end
      
      for jj=1:nann
        iann = ianns(jj);
        a = obj.j.annotations(iann);
        roi = Coco.bbox2roi(a.bbox);
        kps = reshape(a.keypoints,3,npts)';
        if kpsfull 
          % none
        else
          kps = kps(:,1:2);
        end
        
        roiall(jj,:) = roi;
        kpsall(:,:,jj) = kps;
      end
      
      roi = roiall;
      kps = kpsall;
    end
  end
  methods (Static)
    function roi = computeRoi(xy,varargin)
      % See Labeler/maGetRoi
      % 
      % xy: [npts x 2]
      %
      % roi: [4x2] [x(:) y(:)] corners of ractangular roi
      %
      % Note: xy should be ok to be either 0b or 1b, with roi reported
      %   accordingly
      % Note: roi edges are conceptually infinitely thin (vs eg having pixel width)
      %   Therefore roi high-coordinates are not "one past"
     
      
      [iptHead,tfscaled,tfincfixedmargin,scalefac,radfixed] = ...
        myparse(varargin,...
        'iptHead',nan,... % if supplied, alignHT is done
        'scaledToTgt',true,...
        'incFixedMargin',true,...
        'scalefac',1.0,...
        'radfixed',0.0 ...
        );
      
      tfalignHT = ~isnan(iptHead); 
      
      if tfscaled 
        if tfincfixedmargin
          scaledfixedmargin = radfixed;
        else
          scaledfixedmargin = 0;
        end
      end
        
      if tfalignHT
        xyH = xy(iptHead,:);
        xyCent = nanmean(xy,1);

        v = xyH-xyCent; % vec from centroid->head
        phi = atan2(v(2),v(1)); % azimuth of vec from t->h
        R = rotationMatrix(-phi);
        
        xyc = xy-xyCent; % kps centered about centroid
        Rxyc = R*xyc.'; % [2xnpts] centered, rotated kps
                        % vec from cent->h should point to positive x
        if tfscaled
          Rroi = Labeler.maRoiXY2RoiScaled(Rxyc.',scalefac,scaledfixedmargin); 
        else
          Rroi = Labeler.maRoiXY2RoiFixed(Rxyc.',radfixed);
        end
        % Rroi is [4x2]

        roi = R.'*Rroi.'; 
        roi = roi.'+xyCent;
      else
        if tfscaled
          roi = Labeler.maRoiXY2RoiScaled(xy,scalefac,scaledfixedmargin);
        else
          roi = Labeler.maRoiXY2RoiFixed(xy,radfixed);
        end
      end
    end
    function roi = bbox2roi(bb)
      % bb: [4]
      assert(numel(bb)==4);
      bb = bb(:)';
      roi = [bb(1:2) bb(1:2)+bb(3:4)];      
    end
  end
end