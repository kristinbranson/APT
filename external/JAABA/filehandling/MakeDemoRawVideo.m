function MakeDemoRawVideo(moviename,varargin)

defaults.fps = 20;
defaults.compression = 'None';
allowedcompressions = {'Indeo3', 'Indeo5', 'Cinepak', 'MSVC', 'RLE', 'None','Uncompressed AVI','Motion JPEG AVI'};
useVideoWriter = exist('VideoWriter','file');
mencoderoptions = '';
mencoder_maxnframes = inf;
[moviename,aviname,zoomboxloc,...
  fps,maxnframes,firstframes,firstframefracs,...
  compression,figpos,movietitle,...
  useVideoWriter,mencoderoptions,mencoder_maxnframes,...
  avifileTempDataFile,titletext,doflipud,dofliplr] = ...
  myparse(varargin,'moviename','aviname','','zoomboxloc',[],...
  'fps',nan,'maxnframes',nan,'firstframes',[],'firstframefracs',[],...
  'compression','','figpos',[],'movietitle','','useVideoWriter',useVideoWriter,...
  'mencoderoptions',mencoderoptions,'mencoder_maxnframes',mencoder_maxnframes,...
  'avifileTempDataFile','',...
  'titletext',true,...
  'flipud',false,'fliplr',false);

if ~ischar(compression),
  compression = '';
end
if ~isempty(compression) && ~any(strcmpi(compression,allowedcompressions)),
  fprintf('Unknown compressor %s\n',compression);
  compression = '';
end

if ~ischar(moviename) || isempty(moviename) || ~exist(moviename,'file'),
  fprintf('Choose raw movie to annotate\n');
  helpmsg = 'Choose raw movie to annotate';
  [movienameonly,moviepath] = uigetfilehelp({'*.fmf';'*.sbfmf';'*.avi'},'Choose raw movie to annotate','','helpmsg',helpmsg);
  if ~ischar(movienameonly),
    return;
  end
  moviename = [moviepath,movienameonly];
else
  [moviepath,movienameonly] = split_path_and_filename(moviename);
end
[readframe1,nframes,fid] = get_readframe_fcn(moviename);
if doflipud && dofliplr,
  readframe = @(x) flipdim(flipdim(readframe1(x),1),2);
elseif doflipud,
  readframe = @(x) flipdim(readframe1(x),1);
elseif dofliplr
  readframe = @(x) flipdim(readframe1(x),2);
else
  readframe = readframe1;
end
  
if fid < 0,
  uiwait(msgbox(sprintf('Could not read in movie %s',moviename)));
  return;
end

% output avi
haveaviname = false;
if ischar(aviname) && ~isempty(aviname)
  [~,ext] = splitext(aviname);
  if strcmpi(ext,'.avi'),
    haveaviname = true;
  end
end
if ~haveaviname,
  fprintf('Choose avi file to output annotated version of %s\n',moviename);
  [base,~] = splitext(movienameonly);
  aviname = [moviepath,'ctraxresults_',base,'.avi'];
  helpmsg = {};
  helpmsg{1} = 'Choose avi file to write annotated movie to.';
  helpmsg{2} = sprintf('Raw movie input: %s',moviename);
  helpmsg{3} = sprintf('Trx file name input: %s',trxname);
  [avinameonly,avipath] = uiputfilehelp('*.avi',sprintf('Choose output avi for %s',movienameonly),aviname,'helpmsg',helpmsg);
  if ~ischar(avinameonly),
    return;
  end
  aviname = [avipath,avinameonly];
end

if isempty(zoomboxloc),
  if ishandle(1),
    close(1);
  end
  figure(1);
  clf;
  imagesc(readframe(round(nframes/2)));
  axis image;
  while true,
    title('Click and drag to draw rectangular region to zoom in on');
    res = getrect(1);
    title('');
    zoomboxloc = [round(res(1)),round(res(1)+res(3)),round(res(2)),round(res(2)+res(4))];
    h = plot(zoomboxloc([1,2,2,1,1]),zoomboxloc([3,3,4,4,3]),'r.-');
    res = questdlg('Happy with zoom box?','Happy?');
    if strcmpi(res,'Cancel'),
      return;
    end
    if strcmpi(res,'Yes'),
      break;
    end
    delete(h);
  end
end

if isempty(firstframes) && ~isempty(firstframefracs),
  firstframes = min(nframes,max(1,round(firstframefracs*(nframes-1)+1)));
end
  
prompts = {};
defaultanswers = {};
if isnan(fps) && isfield(headerinfo,'timestamps'),
  fps = 1/median(diff(headerinfo.timestamps));
end
if isnan(fps),
  prompts{end+1} = 'Output movie frames per second';
  defaultanswers{end+1} = num2str(defaults.fps);
end
if isnan(maxnframes),
  prompts{end+1} = 'Max number of frames to output';
  defaultanswers{end+1} = num2str(500);
end
if isempty(firstframes),
  prompts{end+1} = 'First frame to output';
  defaultanswers{end+1} = num2str(1);
end

compressionprompt = ['Compressor (must be one of ',...
  sprintf('%s, ',allowedcompressions{1:end-1}),allowedcompressions{end},')'];
if ~ispc && ~useVideoWriter,
  compression = 'None';
elseif isempty(compression),
  prompts{end+1} = compressionprompt;
  defaultanswers{end+1} = defaults.compression;
end

if ~isempty(prompts),
  while true,
    answers = inputdlg(prompts,'make_ctrax_result_movie parameters',1,defaultanswers);
    if isempty(answers),
      return;
    end
    failed = false;
    for i = 1:length(answers),
      if strcmp(prompts{i},compressionprompt)
        j = strmatch(answers{i},allowedcompressions);
        if isempty(j),
          uiwait(msgbox(sprintf('Illegal compressor: %s',answers{i})));
          failed = true;
          break;
        end
        compression = allowedcompressions{j};
        answers{i} = allowedcompressions{j};
        defaultanswers{i} = compression;
        continue;
      end
      answers{i} = str2double(answers{i});
      if isempty(answers{i}) || answers{i} < 0,
        uiwait(msgbox('All answers must be positive numbers'));
        failed = true;
        break;
      end
      switch prompts{i},
        case 'Output movie frames per second',
          fps = answers{i};
          answers{i} = fps;
        case 'Max number of frames to output',
          maxnframes = min(ceil(answers{i}),nframes);
          answers{i} = maxnframes;
        case 'First frame to output',
          firstframes = max(1,ceil(answers{i}));
          answers{i} = firstframes;
      end
      defaultanswers{i} = num2str(answers{i});
    end
    if failed,
      continue;
    end
    break;
  end
end

endframes = min(nframes,firstframes+maxnframes-1);
im = readframe(firstframes(1));
[nr,nc,ncolors] = size(im);

if ishandle(1),
  close(1);
end
figure(1);
clf;
aspectratio_whole = 
hold on;
hax = gca;
set(hax,'position',[0,0,1,1]);
axis off;
isdisplay = ispc || ~strcmpi(get(1,'XDisplay'),'nodisplay');

% corners of zoom boxes in plotted image coords
x0 = nc+(0:nzoomc-1)*rowszoom+1;
y0 = (0:nzoomr-1)*rowszoom+1;
x1 = x0 + rowszoom - 1;
y1 = y0 + rowszoom - 1;

% relative frame offset
nframesoff = getstructarrayfield(trx,'firstframe') - 1;

% pre-allocate
himzoom = zeros(nzoomr,nzoomc);
htail = zeros(1,nids);
htri = zeros(1,nids);
hwing = zeros(1,nids);
hsexmarker = zeros(1,nids);
scalefactor = rowszoom / (2*boxradius+1);
hzoom = zeros(nzoomr,nzoomc);
hzoomwing = zeros(nzoomr,nzoomc);
htextzoom = zeros(nzoomr,nzoomc);

mencoder_nframes = 0;
tic;

for segi = 1:numel(firstframes),
  firstframe = firstframes(segi);
  endframe = endframes(segi);

  for frame = firstframe:endframe,
    if mod(frame - firstframe,5) == 0,
      fprintf('frame %d, write rate = %f s/fr\n',frame,toc/5);
      tic;
    end
    
    % relative frame
    idx = frame - nframesoff;
    
    isalive = frame >= getstructarrayfield(trx,'firstframe') & ...
      frame <= getstructarrayfield(trx,'endframe');
    
    % draw the unzoomed image
    im = uint8(readframe(frame));
    if ncolors == 1,
      im = repmat(im,[1,1,3]);
    end
    if frame == firstframes(1),
      him = image([1,nc],[1,nr],im);
      axis image;
      axis([.5,x1(end)+.5,.5,y1(end)+.5]);
      axis off;
    else
      set(him,'cdata',im);
    end
    
    % draw frame number text box
    framestr = sprintf('Frame %d, t = %.2f s',frame,timestamps(frame)-timestamps(1));
    if ~isempty(movietitle),
      framestr = {framestr,movietitle}; %#ok<AGROW>
    end
    % text doesn't show up in no display mode
    if titletext && isdisplay,
      if frame == firstframes(1),
        htext = text(.5,.5,framestr,'Parent',hax,'BackgroundColor','k','Color','g','VerticalAlignment','bottom','interpreter','none');
      else
        set(htext,'String',framestr);
      end
    end
    
    % draw the zoomed image
    if dynamicflyselection && exist('fliesplotperframe','var'),
      zoomflies = reshape(fliesplotperframe(:,frame),[nzoomr,nzoomc]);
    end
    for i = 1:nzoomr,
      for j = 1:nzoomc,
        fly = zoomflies(i,j);
        
        % fly not visible?
        if isnan(fly) || ~isalive(fly),
          if frame == firstframes(1),
            himzoom(i,j) = image([x0(j),x1(j)],[y0(i),y1(i)],repmat(uint8(123),[boxradius*2+1,boxradius*2+1,3]));
          else
            set(himzoom(i,j),'cdata',repmat(uint8(123),[boxradius*2+1,boxradius*2+1,3]));
          end
          continue;
        end
        
        % grab a box around (x,y)
        x = round(trx(fly).x(idx(fly)));
        y = round(trx(fly).y(idx(fly)));
        boxradx1 = min(boxradius,x-1);
        boxradx2 = min(boxradius,size(im,2)-x);
        boxrady1 = min(boxradius,y-1);
        boxrady2 = min(boxradius,size(im,1)-y);
        box = uint8(zeros(2*boxradius+1));
        box(boxradius+1-boxrady1:boxradius+1+boxrady2,...
          boxradius+1-boxradx1:boxradius+1+boxradx2) = ...
          im(y-boxrady1:y+boxrady2,x-boxradx1:x+boxradx2);
        if frame == firstframes(1),
          himzoom(i,j) = image([x0(j),x1(j)],[y0(i),y1(i)],repmat(box,[1,1,3]));
        else
          set(himzoom(i,j),'cdata',repmat(box,[1,1,3]));
        end
        
      end
    end;
    
    % plot the zoomed out position
    if frame == firstframes(1),
      for fly = 1:nids,
        if isalive(fly),
          i0 = max(1,idx(fly)-taillength);
          htail(fly) = plot(trx(fly).x(i0:idx(fly)),trx(fly).y(i0:idx(fly)),...
            '-','color',colors(fly,:));
          if doshowsex,
            sexi = find(strcmpi(trx(fly).sex{idx(fly)},sexes),1);
            sexmarker = sexmarkers{sexi};
            hsexmarker(fly) = plot(trx(fly).x(idx(fly)),trx(fly).y(idx(fly)),'.',...
              'color',colors(fly,:),'marker',sexmarker,'markerfacecolor',colors(fly,:));
          end
          htri(fly) = drawflyo(trx(fly),idx(fly));
          set(htri(fly),'color',colors(fly,:));
          if doplotwings,
            xwing = [trx(fly).xwingl(idx(fly)),trx(fly).x(idx(fly)),trx(fly).xwingr(idx(fly))];
            ywing = [trx(fly).ywingl(idx(fly)),trx(fly).y(idx(fly)),trx(fly).ywingr(idx(fly))];
            hwing(fly) = plot(xwing,ywing,'.-','color',colors(fly,:));
          end
        else
          htail(fly) = plot(nan,nan,'-','color',colors(fly,:));
          if doshowsex,
            hsexmarker(fly) = plot(nan,nan,'.','color',colors(fly,:),'markerfacecolor',colors(fly,:));
          end
          htri(fly) = plot(nan,nan,'-','color',colors(fly,:));
          if doplotwings,
            hwing(fly) = plot(nan,nan,'.-','color',colors(fly,:));
          end
        end
      end
    else
      for fly = 1:nids,
        if isalive(fly),
          i0 = max(1,idx(fly)-taillength);
          set(htail(fly),'xdata',trx(fly).x(i0:idx(fly)),...
            'ydata',trx(fly).y(i0:idx(fly)));
          if doshowsex,
            sexi = find(strcmpi(trx(fly).sex{idx(fly)},sexes),1);
            sexmarker = sexmarkers{sexi};
            set(hsexmarker(fly),'xdata',trx(fly).x(idx(fly)),...
              'ydata',trx(fly).y(idx(fly)),...
              'color',colors(fly,:),...
              'marker',sexmarker,...
              'markerfacecolor',colors(fly,:));
          end
          updatefly(htri(fly),trx(fly).x(idx(fly)),trx(fly).y(idx(fly)),...
            trx(fly).theta(idx(fly)),trx(fly).a(idx(fly)),trx(fly).b(idx(fly)));
          if doplotwings,
            xwing = [trx(fly).xwingl(idx(fly)),trx(fly).x(idx(fly)),trx(fly).xwingr(idx(fly))];
            ywing = [trx(fly).ywingl(idx(fly)),trx(fly).y(idx(fly)),trx(fly).ywingr(idx(fly))];
            set(hwing(fly),'XData',xwing,'YData',ywing);
          end
        else
          set(htail(fly),'xdata',[],'ydata',[]);
          set(htri(fly),'xdata',[],'ydata',[]);
          if doshowsex,
            set(hsexmarker(fly),'xdata',[],'ydata',[]);
          end
          if doplotwings,
            set(hwing(fly),'XData',[],'YData',[]);
          end
        end
      end
    end
    
    % plot the zoomed views
    for i = 1:nzoomr,
      for j = 1:nzoomc,
        fly = zoomflies(i,j);
        if ~isnan(fly) && isalive(fly),
          x = trx(fly).x(idx(fly));
          y = trx(fly).y(idx(fly));
          x = boxradius + (x - round(x))+.5;
          y = boxradius + (y - round(y))+.5;
          x = x * scalefactor;
          y = y * scalefactor;
          x = x + x0(j) - 1;
          y = y + y0(i) - 1;
          a = trx(fly).a(idx(fly))*scalefactor;
          b = trx(fly).b(idx(fly))*scalefactor;
          theta = trx(fly).theta(idx(fly));
          if doshowsex,
            s = sprintf('%d, %s',fly,trx(fly).sex{idx(fly)});
          else
            s = sprintf('%d',fly);
          end
          if doplotwings,
            xwingl = trx(fly).xwingl(idx(fly)) - round(trx(fly).x(idx(fly))) + boxradius + .5;
            ywingl = trx(fly).ywingl(idx(fly)) - round(trx(fly).y(idx(fly))) + boxradius + .5;
            xwingl = xwingl * scalefactor;
            ywingl = ywingl * scalefactor;
            xwingl = xwingl + x0(j) - 1;
            ywingl = ywingl + y0(i) - 1;
            xwingr = trx(fly).xwingr(idx(fly)) - round(trx(fly).x(idx(fly))) + boxradius + .5;
            ywingr = trx(fly).ywingr(idx(fly)) - round(trx(fly).y(idx(fly))) + boxradius + .5;
            xwingr = xwingr * scalefactor;
            ywingr = ywingr * scalefactor;
            xwingr = xwingr + x0(j) - 1;
            ywingr = ywingr + y0(i) - 1;
            xwing = [xwingl,x,xwingr];
            ywing = [ywingl,y,ywingr];
          end

          if frame == firstframes(1),
            hzoom(i,j) = drawflyo(x,y,theta,a,b);
            if doplotwings,
              hzoomwing(i,j) = plot(xwing,ywing,'.-','color',colors(fly,:));
            end
            if isdisplay,
              htextzoom(i,j) = text((x0(j)+x1(j))/2,.95*y0(i)+.05*y1(i),s,...
                'color',colors(fly,:),'horizontalalignment','center',...
                'verticalalignment','bottom','fontweight','bold');
            else
              if doshowsex,
                sexi = find(strcmpi(trx(fly).sex{idx(fly)},sexes),1);
                sexmarker = sexmarkers{sexi};
                htextzoom(i,j) = plot(x,y,'.',...
                  'color',colors(fly,:),'marker',sexmarker,'markerfacecolor',colors(fly,:));
              end
            end
          else
            updatefly(hzoom(i,j),x,y,theta,a,b);
            if doplotwings,
              set(hzoomwing(i,j),'XData',xwing,'YData',ywing,'Color',colors(fly,:));
            end
            if isdisplay,
              set(htextzoom(i,j),'string',s,'color',colors(fly,:));
            else
              if doshowsex,
                sexi = find(strcmpi(trx(fly).sex{idx(fly)},sexes),1);
                sexmarker = sexmarkers{sexi};
                set(htextzoom(i,j),'xdata',x,...
                  'ydata',y,...
                  'color',colors(fly,:),...
                  'marker',sexmarker,...
                  'markerfacecolor',colors(fly,:));
              end
            end
          end
          set(hzoom(i,j),'color',colors(fly,:));
        else
          if frame == firstframes(1),
            hzoom(i,j) = plot(nan,nan,'-');
            if doplotwings,
              hzoomwing(i,j) = plot(nan,nan,'.-');
            end
            if isdisplay,
              htextzoom(i,j) = text((x0(j)+x1(j))/2,.95*y0(i)+.05*y1(i),'',...
                'horizontalalignment','center',...
                'verticalalignment','bottom','fontweight','bold');
            else
              if doshowsex,
                htextzoom(i,j) = plot(nan,nan,'.','marker','none');
              end
            end
          else
            set(hzoom(i,j),'xdata',[],'ydata',[]);
            if doplotwings,
              set(hzoomwing(i,j),'XData',[],'YData',[]);
            end
            if isdisplay,
              set(htextzoom(i,j),'string','');
            else
              if doshowsex,
                set(htextzoom(i,j),'xdata',[],'ydata',[]);
              end
            end
          end
        end
      end
    end
    
    if frame == firstframes(1),
      if ~isempty(figpos),
        set(1,'Position',figpos);
      else
        input('Resize figure 1 to the desired size, hit enter when done.');
        figpos = get(1,'Position');
      end
      set(1,'visible','off');
      if useVideoWriter,
        if strcmpi(compression,'None') || strcmpi(compression,'Uncompressed AVI'),
          profile = 'Uncompressed AVI';
        else
          profile = 'Motion JPEG AVI';
        end
        aviobj = VideoWriter(aviname,profile); %#ok<TNMLP>
        set(aviobj,'FrameRate',fps);
        if ~strcmpi(profile,'Uncompressed AVI'),
          set(aviobj,'Quality',100);
        end
        open(aviobj);
      else
        if isempty(avifileTempDataFile),
          aviobj = avifile(aviname,'fps',fps,'quality',100,'compression',compression);  %#ok<REMFF1>
        else
          aviobj = myavifile(aviname,'fps',fps,'quality',100,'compression',compression,...
            'TempDataFile',avifileTempDataFile); 
          fprintf('Temporary data file for avi writing: %s\n',aviobj.TempDataFile);
        end
      end
    end
    
    if frame == firstframes(1),
      fr = getframe_invisible(hax);
      [height,width,~] = size(fr);
      fprintf('Size of frame is %d x %d\n',height,width);
      gfdata = getframe_initialize(hax);
      [fr,height,width] = getframe_invisible_nocheck(gfdata,[height,width],false,false);

%       height = ceil(height/4)*4;
%       width = ceil(width/4)*4;
%       fr = getframe_invisible(hax,[height,width]);
    else
      fr = getframe_invisible_nocheck(gfdata,[height,width],false);
    end
    if useVideoWriter,
      writeVideo(aviobj,fr);
    else
      aviobj = addframe(aviobj,fr);
    end
    set(1,'Position',figpos);
    
  end
  
end
  
fprintf('Finishing AVI...\n');

if useVideoWriter,
  close(aviobj);
else
  aviobj = close(aviobj); %#ok<NASGU>
end
if fid > 0,
  fclose(fid);
end

fprintf('Cleanup...\n');

getframe_cleanup(gfdata);

succeeded = true;



