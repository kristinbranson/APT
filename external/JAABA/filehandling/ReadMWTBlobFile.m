function trx = ReadMWTBlobFile(filename,trx,id)

% multiple files input, then call recursively
if iscell(filename),
  if nargin < 2,
    trx = [];
  end
  if nargin < 3,
    id = [];
  end
  for i1 = 1:numel(filename),
    trx = ReadMWTBlobFile(filename{i1},trx,id);
  end
  return;
end

nparams = 10;
nspinepts = 11;
contouroffset = '0';

if nargin < 2 || isempty(trx),
  trx = [];
  ids = [];
  havereadcontour = false;
  havereadspine = false;
else
  ids = [trx.id];
  havereadcontour = isfield(trx,'xcontour');
  havereadspine = isfield(trx,'xspine');
end

blobsext = '.blobs';
[~,~,ext] = fileparts(filename);
isblobs = strcmp(blobsext,ext);

fid = fopen(filename,'r');
if fid < 1,
  error('Could not open file %s for reading',filename);
end

% read in the id
if isblobs,
  while true,
    id = nan;
    isdone = false;
    while true,
      l1 = fgetl(fid);
      if ~ischar(l1),
        isdone = true;
        break;
      end
      if isempty(l1),
        continue;
      end
      if l1(1) == '%',
        id = str2double(l1(2:end));
      end
      break;
    end
    if isdone,
      break;
    end
    if isnan(id),
      warning('Could not parse id for line >%s<',l1);
      continue;
    end
    ReadMWTBlobFile1();
  end
else
  if nargin < 3 || isempty(id),
    id = 1;
  end
  ReadMWTBlobFile1();
end
% 
% % set nframes
% for i1 = 1:numel(trx),
%   trx(i1).nframes = trx(i1).endframe - trx(i1).firstframe + 1;
% end

% sort by ids
[ids,order] = sort(ids);
trx = trx(order);

% check for missing data
for i1 = 1:numel(trx),
  if any(isnan(trx(i1).timestamps)),
    warning('%d frames of data not set for target %d',nnz(isnan(trx(i1).timestamps)),i1);
  end
end

fclose(fid);

  function ReadMWTBlobFile1()
    
    idi = find(ids == id,1);
    if ~isempty(idi),
      warning('Reading data for id %d more than once',id);
    end
    
    while true,
      filepos = ftell(fid);
      l = fgetl(fid);
      if ~ischar(l),
        break;
      end
      if isempty(l),
        continue;
      end
      % new id
      if isblobs && l(1) == '%',
        fseek(fid,filepos,'bof');
        break;
      end
      
      % parse the parameters in the data line
      [data_curr,count,errmsg,nextindex] = sscanf(l,'%f',nparams);
      if count < nparams || ~isempty(errmsg),
        warning('Could not parse data for id %d, line >%s<, error message: %s',id,l,errmsg);
        continue;
      end
      l = l(nextindex:end);
      
      % read in the contour, spine if it exists
      readcontour = false;
      readspine = false;
      while ~isempty(l),
        [w,count,~,nextindex] = sscanf(l,'%s',1);
        if count == 1,
          l = l(nextindex:end);
          switch w,
            case '%',
              [tmp,count,~,nextindex] = sscanf(l,'%f',nspinepts*2);
              if count == nspinepts*2,
                readspine = true;
                xspine = tmp(1:2:end-1);
                yspine = tmp(2:2:end);
              else
                warning('Error parsing spine points');
              end
              l = l(nextindex:end);
            case '%%',
              [tmp,count,~,nextindex] = sscanf(l,'%f',3);
              l = l(nextindex:end);
              if count == 3,
                xstart = tmp(1);
                ystart = tmp(2);
                ncontourpts = tmp(3);
                [a,count,~,nextindex] = sscanf(l,'%s',1);
                l = l(nextindex:end);
                if count == 1,
                  % convert to binary
                  b = dec2bin(a - contouroffset,8) == '1';
                  b = b(:,3:end)';
                  % first bit is 1 when y is changing
                  isy = b(1:2:end-1);
                  % second bit is 1 when positive change
                  ispos = b(2:2:end);
                  isy = isy(1:ncontourpts-1);
                  ispos = ispos(1:ncontourpts-1);
                  dx = double(~isy);
                  dx(~ispos) = -dx(~ispos);
                  dy = double(isy);
                  dy(~ispos) = -dy(~ispos);
                  xcontour = cumsum([xstart,dx]);
                  ycontour = cumsum([ystart,dy]);
                  readcontour = true;
                end
              end
            otherwise,
              warning('Unknown delimiter %s',w);
          end
        end
      end
      
      if ~readcontour && havereadcontour,
        xcontour = {};
        ycontour = {};
      end
      if ~readspine && havereadspine,
        xspine = nan(nspinepts,1);
        yspine = nan(nspinepts,1);
      end
      
      f = data_curr(1);
      timestamps = data_curr(2);
      x = data_curr(3);
      y = data_curr(4);
      area = data_curr(5);
      vx = data_curr(6);
      vy = data_curr(7);
      a = sqrt(vx^2+vy^2);
      theta = atan2(vy,vx);
      b = data_curr(8);
      length = data_curr(9);
      width = data_curr(10);
      
      if readspine,
        xspine = xspine + x;
        yspine = yspine + y;
      end
      
      % first time reading a contour
      if readcontour && ~havereadcontour,
        for tmpi = 1:numel(trx),
          nframescurr = trx(tmpi).endframe - trx(tmpi).firstframe + 1;
          trx(tmpi).xcontour = cell(1,nframescurr);
          trx(tmpi).ycontour = cell(1,nframescurr);
        end
      end
      % first time reading a spine
      if readspine && ~havereadspine,
        for tmpi = 1:numel(trx),
          nframescurr = trx(tmpi).endframe - trx(tmpi).firstframe + 1;
          trx(tmpi).xspine = nan(nspinepts,nframescurr);
          trx(tmpi).yspine = nan(nspinepts,nframescurr);
        end
      end
      
      if isempty(idi),
        idi = numel(ids) + 1;
        newtrx = struct('x',x,'y',y,'theta',theta,'a',a,'b',b,'id',id,...
          'firstframe',f,'nframes',nan,'endframe',f,'off',1-f,'timestamps',timestamps,...
          'area',area,'length',length,'width',width);
        if readcontour,
          newtrx.xcontour = {xcontour(:)};
          newtrx.ycontour = {ycontour(:)};
        elseif havereadcontour,
          newtrx.xcontour = {};
          newtrx.ycontour = {};
        end
        if readspine,
          newtrx.xspine = xspine(:);
          newtrx.yspine = yspine(:);
        elseif havereadspine,
          newtrx.xspine = nan(nspinepts,1);
          newtrx.yspine = nan(nspinepts,1);
        end
                
        if isempty(ids),
          trx = newtrx;
        else
          trx = structarrayset(trx,idi,newtrx);
        end
        ids(idi) = id;
        
      else
        
        
        % assume everything comes in order of frames, but some frames may
        % be skipped
        npad = f-trx(idi).endframe-1;
        if npad > 0,
          trx(idi).x(end+1:end+npad) = nan;
          trx(idi).y(end+1:end+npad) = nan;
          trx(idi).a(end+1:end+npad) = nan;
          trx(idi).b(end+1:end+npad) = nan;
          trx(idi).theta(end+1:end+npad) = nan;
          trx(idi).length(end+1:end+npad) = nan;
          trx(idi).width(end+1:end+npad) = nan;
          trx(idi).area(end+1:end+npad) = nan;
          trx(idi).timestamps(end+1:end+npad) = nan;
          
          if readcontour || havereadcontour,
            tmp = cell(1,npad);
            trx(idi).xcontour = [trx(idi).xcontour,tmp];
            trx(idi).ycontour = [trx(idi).ycontour,tmp];
          end
          
          if readspine || havereadspine,
            trx(idi).xspine(:,end+1:end+npad) = nan;
            trx(idi).yspine(:,end+1:end+npad) = nan;
          end
        end
          
        i = f - trx(idi).firstframe + 1;
        trx(idi).x(i) = x;
        trx(idi).y(i) = y;
        trx(idi).a(i) = a;
        trx(idi).b(i) = b;
        trx(idi).theta(i) = theta;
        trx(idi).length(i) = length;
        trx(idi).width(i) = width;
        trx(idi).area(i) = area;
        trx(idi).timestamps(i) = timestamps;
        
        if readcontour || havereadcontour,
          trx(idi).xcontour{i} = xcontour;
          trx(idi).ycontour{i} = ycontour;
        end
        
        if readspine || havereadspine,
          trx(idi).xspine(:,i) = xspine;
          trx(idi).yspine(:,i) = yspine;
        end
        trx(idi).endframe = max(f,trx(idi).endframe);

      end
              
      havereadcontour = havereadcontour || readcontour;
      havereadspine = havereadspine || readspine;
            
    end
    
    if isempty(idi),
      %fprintf('Read 0 frames for id %d\n',id);
    else
      trx(idi).nframes = trx(idi).endframe-trx(idi).firstframe+1;
      %fprintf('Read %d frames for id %d\n',trx(idi).nframes,id);
    end
    
    if havereadcontour,
      for tmpi = 1:numel(trx),
        nframescurr = trx(tmpi).nframes;
        nadd = nframescurr - numel(trx(tmpi).xcontour);
        if nadd > 0,
          trx(tmpi).xcontour(end+1:end+nadd) = cell(1,nadd);
          trx(tmpi).ycontour(end+1:end+nadd) = cell(1,nadd);
        end
      end
    end
    if havereadspine,
      for tmpi = 1:numel(trx),
        nframescurr = trx(tmpi).nframes;
        nadd = nframescurr - size(trx(tmpi).xspine,2);
        if nadd > 0,
          trx(tmpi).xspine(:,end+1:end+nadd) = nan;
          trx(tmpi).yspine(:,end+1:end+nadd) = nan;
        end
      end
    end
    
  end

end
