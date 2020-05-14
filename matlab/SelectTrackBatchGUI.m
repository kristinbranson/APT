classdef SelectTrackBatchGUI < handle

  properties
    toTrack = [];
    lObj = [];
    hParent = [];
    nmovies = 0;
    page = 1;
    npages = 1;
    posinfo = struct;
    gdata = struct;
    nmovies_per_page = 10;
  end
  
  methods
    function obj = SelectTrackBatchGUI(lObj,varargin)

      obj.lObj = lObj;
      obj.hParent = obj.lObj.gdata.figure;
      
      toTrack = myparse(varargin,'toTrack',struct);
      
      obj.initData(toTrack);
      
      obj.createGUI();
      
    end
    
    function initData(obj,toTrack)
      obj.toTrack = toTrack;
      if ~isfield(obj.toTrack,'movfiles'),
        obj.toTrack.movfiles = {};
      end
      if ~isfield(obj.toTrack,'trkfiles'),
        obj.toTrack.trkfiles = {};
      end
      if ~isfield(obj.toTrack,'trxfiles'),
        obj.toTrack.trxfiles = {};
      end
      if ~isfield(obj.toTrack,'cropRois'),
        obj.toTrack.cropRois = {};
      end
      if ~isfield(obj.toTrack,'calibrationfiles'),
        obj.toTrack.calibrationfiles = {};
      end
      if ~isfield(obj.toTrack,'targets'),
        obj.toTrack.targets = {};
      end
      if ~isfield(obj.toTrack,'f0s'),
        obj.toTrack.f0s = {};
      end
      if ~isfield(obj.toTrack,'f1s'),
        obj.toTrack.f1s = {};
      end
      obj.nmovies = size(obj.toTrack.movfiles,1);
    end
    
    function createGUI(obj)
      
      figname = 'Select movies to track';
      
      % compute the figure size
      units = get(obj.hParent,'units');
      set(obj.hParent,'Units','normalized');
      mainfigpos = get(obj.hParent,'Position');
      set(obj.hParent,'Units',units);
      figsz = [.4,.4]; % width, height
      mainfigctr = mainfigpos([1,2]) + mainfigpos([3,4])/2;
      figpos = [mainfigctr-figsz/2,figsz];
      % to do: make sure this is on the screen
      
      border = .025;
      %colw = .3;
      filebuttonw = .05;
      rowh = .05;
      rowborder = .01;
      colborder = .005;
      backgroundcolor = [0,0,0];
      editfilecolor = [.2,.2,.2];
      deletebuttoncolor = [.7,0,0];
      pagebuttoncolor = [0,0,0];
      addbuttoncolor = [.7,0,.7];
      
      colw = ((1-2*border) - (filebuttonw+colborder)*2)/2;
      
      allmovieh = 1 - 3*border - rowh*3 - rowborder - border;
      obj.nmovies_per_page = floor(allmovieh/(rowh+rowborder))-1;
      obj.setNPages();
      moveditx = border;
      coltitley = 1-border-rowh;
      trkeditx = moveditx + colw + colborder;
      detailsbuttonx = trkeditx+colw+colborder;
      deletebuttonx = detailsbuttonx+filebuttonw+colborder;
      rowys = coltitley - (rowh+rowborder)*(1:obj.nmovies_per_page);
      
      pagetextw = .1;
      pagebuttonstrs = {'|<','<','>','>|'};
      npagebuttons = numel(pagebuttonstrs);
      allpagebuttonws = npagebuttons*filebuttonw + npagebuttons*colborder + pagetextw;
      pagebuttonx1 = .5-allpagebuttonws/2;
      pagebuttonxsless = pagebuttonx1 + (filebuttonw+colborder)*[0,1];
      pagetextx = pagebuttonxsless(end)+filebuttonw+colborder;
      pagebuttonxsmore = pagetextx + pagetextw + colborder+(filebuttonw+colborder)*[0,1];
      pagebuttonxs = [pagebuttonxsless,pagebuttonxsmore];
      pagebuttony = rowys(end) - (rowh+rowborder);
      
      addbuttonw = .15;
      addbuttonx = .5 - addbuttonw/2;
      addbuttony = pagebuttony - rowh - border;
      
      % save, track, cancel
      controlbuttonw = .15;
      controlbuttonstrs = {'Save...','Load...','Track...','Cancel'};
      controlbuttontags = {'save','load','track','cancel'};
      controlbuttoncolors = ...
        [0,0,.8
        0,.7,.7
        0,.7,0
        .7,.7,0];
      ncontrolbuttons = numel(controlbuttonstrs);
      allcontrolbuttonw = ncontrolbuttons*controlbuttonw + (ncontrolbuttons-1)*colborder;
      controlbuttonx1 = .5-allcontrolbuttonw/2;
      controlbuttonxs = controlbuttonx1 + (controlbuttonw+colborder)*(0:ncontrolbuttons-1);
      controlbuttony = border;
      
      defaultmovfiles = cell(1,obj.nmovies_per_page);
      defaulttrkfiles = cell(1,obj.nmovies_per_page);
      for i = 1:obj.nmovies_per_page,
        defaultmovfiles{i} = sprintf('pathtomovie%disveryveryvery/long/movieloc%d.avi',i,i);
        defaulttrkfiles{i} = sprintf('pathtotrk%disveryveryvery/long/outputtrkloc%d.trk',i,i);
      end
      obj.gdata = struct;
      
      obj.gdata.fig = figure(...
        'menubar','none',...
        'toolbar','none',...
        'name',figname,...
        'NumberTitle','off',...
        'IntegerHandle','off',...
        'Tag','figure_SelectTrackBatch',...
        'color',backgroundcolor,...
        'units','normalized',...
        'position',figpos);
      %  'windowstyle','modal',...
      
      obj.gdata.txt_movietitle = uicontrol('Style','text','String','Movie',...
        'ForegroundColor','w','BackgroundColor','k','FontWeight','bold',...
        'Units','normalized','Position',[moveditx,coltitley,colw,rowh],...
        'Tag','Movie title');
      obj.gdata.txt_trktitle = uicontrol('Style','text','String','Output trk',...
        'ForegroundColor','w','BackgroundColor','k','FontWeight','bold',...
        'Units','normalized','Position',[trkeditx,coltitley,colw,rowh],...
        'Tag','Trk title');
      for i = 1:obj.nmovies_per_page,
        visible = 'off';
        movfilecurr = defaultmovfiles{i};
        trkfilecurr = defaulttrkfiles{i};
        obj.gdata.edit_movie(i) = uicontrol('Style','edit','String',movfilecurr,...
          'ForegroundColor','w','BackgroundColor',editfilecolor,'FontWeight','normal',...
          'Units','normalized','Enable','on','Position',[moveditx,rowys(i),colw,rowh],...
          'Tag',sprintf('edit_movie%d',i),'UserData',i,...
          'HorizontalAlignment','right','Visible',visible,...
          'Callback',@(h,e) obj.edit_movie_Callback(h,e,i));
        obj.gdata.edit_trk(i) = uicontrol('Style','edit','String',trkfilecurr,...
          'ForegroundColor','w','BackgroundColor',editfilecolor,'FontWeight','normal',...
          'Units','normalized','Enable','on','Position',[trkeditx,rowys(i),colw,rowh],...
          'Tag',sprintf('edit_trk%d',i),'UserData',i,...
          'HorizontalAlignment','right','Visible',visible,...
          'Callback',@(h,e) obj.edit_trk_Callback(h,e,i));
        obj.gdata.button_details(i) = uicontrol('Style','pushbutton','String','...',...
          'ForegroundColor','w','BackgroundColor',editfilecolor,'FontWeight','normal',...
          'Units','normalized','Enable','on','Position',[detailsbuttonx,rowys(i),filebuttonw,rowh],...
          'String','...','Tag',sprintf('pushbutton_details%d',i),'UserData',i,...
          'Visible',visible,...
          'Callback',@(h,e) obj.pb_details_Callback(h,e,i));
        obj.gdata.button_delete(i) = uicontrol('Style','pushbutton','String','-',...
          'ForegroundColor','w','BackgroundColor',deletebuttoncolor,'FontWeight','bold',...
          'Units','normalized','Enable','on','Position',[deletebuttonx,rowys(i),filebuttonw,rowh],...
          'String','-','Tag',sprintf('pushbutton_delete%d',i),'UserData',i,...
          'ToolTip','Remove this movie from list',...
          'Visible',visible,...
          'Callback',@(h,e) obj.pb_delete_Callback(h,e,i));
      end
      
      for i = 1:npagebuttons,
        obj.gdata.button_page(i) = uicontrol('Style','pushbutton','String',pagebuttonstrs{i},...
          'ForegroundColor','w','BackgroundColor',pagebuttoncolor,'FontWeight','bold',...
          'Units','normalized','Enable','on','Position',[pagebuttonxs(i),pagebuttony,filebuttonw,rowh],...
          'Tag',sprintf('pushbutton_page%d',i),'UserData',i,...
          'Callback',@(h,e) obj.pb_page_Callback(h,e,pagebuttonstrs{i}));
      end
      obj.gdata.text_page = uicontrol('Style','text','String',sprintf('Page %d/%d',obj.page,obj.npages),...
        'ForegroundColor','w','BackgroundColor',backgroundcolor,'FontWeight','normal',...
        'Units','normalized','Enable','on','Position',[pagetextx,pagebuttony,pagetextw,rowh],...
        'Tag','text_page','HorizontalAlignment','center');
      obj.gdata.button_add = uicontrol('Style','pushbutton','String','Add movie...',...
        'ForegroundColor','w','BackgroundColor',addbuttoncolor,'FontWeight','bold',...
        'Units','normalized','Enable','on','Position',[addbuttonx,addbuttony,addbuttonw,rowh],...
        'Tag','pushbutton_add',...
        'Callback',@(h,e) obj.pb_add_Callback(h,e));
      
      for i = 1:ncontrolbuttons,
        obj.gdata.button_control(i) = uicontrol('Style','pushbutton','String',controlbuttonstrs{i},...
          'ForegroundColor','w','BackgroundColor',controlbuttoncolors(i,:),'FontWeight','bold',...
          'Units','normalized','Enable','on','Position',[controlbuttonxs(i),controlbuttony,controlbuttonw,rowh],...
          'Tag',sprintf('controlbutton_%s',controlbuttontags{i}),'UserData',i,...
          'Callback',@(h,e) obj.pb_control_Callback(h,e,controlbuttontags{i}));
      end
      obj.updateMovieList();

    end
    function updateMovieList(obj)
      
      for i = 1:obj.nmovies_per_page,
        moviei = obj.item2MovieIdx(i);
        if moviei <= obj.nmovies,
          movfilecurr = obj.toTrack.movfiles{moviei,1};
          trkfilecurr = obj.toTrack.trkfiles{moviei,1};
          visible = 'on';
        else
          movfilecurr = '';
          trkfilecurr = '';
          visible = 'off';
        end
        if obj.page == 1 && i == 1,
          visible = 'on';
        end
        set(obj.gdata.edit_movie(i),'String',movfilecurr);
        set(obj.gdata.edit_trk(i),'String',trkfilecurr);
        obj.setRowVisible(i,visible);
        set([obj.gdata.edit_movie(i),obj.gdata.edit_trk(i),...
          obj.gdata.button_details(i),obj.gdata.button_delete(i)],'Visible',visible);
      end
      set(obj.gdata.text_page,'String',sprintf('Page %d/%d',obj.page,obj.npages));
    end
    function setRowVisible(obj,i,visible)
      set([obj.gdata.edit_movie(i),obj.gdata.edit_trk(i),...
        obj.gdata.button_details(i),obj.gdata.button_delete(i)],'Visible',visible);
    end
    
    function moviei = item2MovieIdx(obj,itemi)
      moviei = (obj.page-1)*obj.nmovies_per_page + itemi;
    end
    function itemi = movie2ItemIdx(obj,moviei)
      itemi = moviei - (obj.page-1)*obj.nmovies_per_page;
      if itemi <= 0 || itemi > obj.nmovies_per_page,
        itemi = [];
      end
    end
    function movdata = getMovData(obj,moviei)
      movdata = struct;
      if moviei <= obj.nmovies,
        movdata.movfiles = obj.toTrack.movfiles(moviei,:);
        movdata.trkfiles = obj.toTrack.trkfiles(moviei,:);
        movdata.trxfiles = obj.toTrack.trxfiles(moviei,:);
        movdata.calibrationfiles = obj.toTrack.calibrationfiles{moviei};
        movdata.cropRois = obj.toTrack.cropRois{moviei};
        movdata.targets = obj.toTrack.targets{moviei};
        movdata.f0s = obj.toTrack.f0s{moviei};
        movdata.f1s = obj.toTrack.f1s{moviei};
      end
    end
    function setMovData(obj,moviei,movdata)
      nview = size(obj.lObj.nview);
      obj.toTrack.movfiles(moviei,:) = movdata.movfiles;
      obj.toTrack.trkfiles(moviei,:) = movdata.trkfiles;
      if isfield(movdata,'trxfiles'),
        obj.toTrack.trxfiles(moviei,:) = movdata.trxfiles;
      else
        obj.toTrack.trxfiles(moviei,:) = repmat({''},[1,nview]);
      end
      if isfield(movdata,'calibrationfiles'),
        obj.toTrack.calibrationfiles{moviei} = movdata.calibrationfiles;
      else
        obj.toTrack.calibrationfiles{moviei} = '';
      end
      if isfield(movdata,'cropRois') && ~isempty(movdata.cropRois),
        obj.toTrack.cropRois(moviei,:) = movdata.cropRois;
      else
        obj.toTrack.cropRois(moviei,:) = repmat({[]},[1,nview]);
      end
      if isfield(movdata,'targets'),
        obj.toTrack.targets{moviei} = movdata.targets;
      else
        obj.toTrack.targets{moviei} = [];
      end
      if isfield(movdata,'f0s'),
        obj.toTrack.f0s{moviei} = movdata.f0s;
      else
        obj.toTrack.f0s{moviei} = [];
      end
      if isfield(movdata,'f1s'),
        obj.toTrack.f1s{moviei} = movdata.f1s;
      else
        obj.toTrack.f1s{moviei} = [];
      end
      itemi = obj.movie2ItemIdx(moviei);
      if isempty(itemi),
        return;
      end
      set(obj.gdata.edit_movie(itemi),'String',obj.toTrack.movfiles{moviei,1});
      set(obj.gdata.edit_trk(itemi),'String',obj.toTrack.trkfiles{moviei,1});
      obj.setRowVisible(itemi,'on');
      
    end
    function pb_details_Callback(obj,h,e,itemi)
      moviei = obj.item2MovieIdx(itemi);
      movdata = obj.getMovData(moviei);
      movdetailsobj = SpecifyMovieToTrackGUI(obj.lObj,obj.gdata.fig,movdata);
      movdataout = movdetailsobj.run();
      obj.setMovData(moviei,movdataout);
    end
    function pb_delete_Callback(obj,h,e,itemi)
      moviei = obj.item2MovieIdx(itemi);
      
      obj.toTrack.movfiles(moviei,:) = [];
      obj.toTrack.trkfiles(moviei,:) = [];
      obj.toTrack.trxfiles(moviei,:) = [];
      obj.toTrack.calibrationfiles(moviei) = [];
      obj.toTrack.targets(moviei) = [];
      obj.toTrack.f0s(moviei) = [];
      obj.toTrack.f1s(moviei) = [];
      obj.nmovies = obj.nmovies - 1;
      obj.setNPages();
      if obj.page > obj.npages,
        obj.page = obj.npages;
      end
      obj.updateMovieList();
      
    end
          
    function pb_add_Callback(obj,h,e)
      movdetailsobj = SpecifyMovieToTrackGUI(obj.lObj,obj.gdata.fig);
      movdataout = movdetailsobj.run();
      if ~isfield(movdataout,'movfiles') || isempty(movdataout.movfiles) || ...
          isempty(movdataout.movfiles{1}),
        return;
      end
      obj.nmovies = obj.nmovies + 1;
      moviei = obj.nmovies;
      obj.setMovData(moviei,movdataout);
      obj.setNPages();
      if obj.page < obj.npages,
        obj.page = obj.npages;
        obj.updateMovieList();
      end
    end
    function setNPages(obj)
      obj.npages = ceil(max(1,obj.nmovies) / obj.nmovies_per_page);
    end
    function setPage(obj,page)
      obj.page = page;
      obj.updateMovieList();
    end
    
    function pb_page_Callback(obj,h,e,key)
      switch key,
        case '|<',
          obj.setPage(1);
        case '>|',
          obj.setPage(obj.npages);
        case '<',
          if obj.page > 1,
            obj.setPage(obj.page-1);
          end
        case '>',
          if obj.page < obj.npages,
            obj.setPage(obj.page+1);
          end
      end
    end
    
    function pb_control_Callback(obj,h,e,tag)
      switch tag,
        case 'cancel',
          delete(obj.gdata.fig);
        case {'load','save'},
          obj.loadOrSaveToFile(tag);
        case 'track',
          res = questdlg('Save list of movies to track for reference?');
          if strcmpi(res,'Cancel'),
            return;
          elseif strcmpi(res,'Yes'),
            success = obj.loadOrSaveToFile('save');
            if ~success,
              return;
            end
          end
          trackBatch('lObj',obj.lObj,'toTrack',obj.toTrack);
          delete(obj.gdata.fig);
        otherwise
          error('Callback for %s not implemented',tag);
      end
    end
    
    function success = loadOrSaveToFile(obj,tag)
      persistent lastpath;
      if isempty(lastpath),
        lastpath = '';
      end
      success = false;
      if strcmpi(tag,'load'),
        [filename,pathname] = uigetfile('*.json','Load list of movies to track from file',lastpath);
      else
        [filename,pathname] = uiputfile('*.json','Load list of movies to track from file',lastpath);
      end
      if ~ischar(filename),
        return;
      end
      jsonfile = fullfile(pathname,filename);
      lastpath = jsonfile;
      if strcmpi(tag,'load'),
        try
          toTrack = parseToTrackJSON(jsonfile,obj.lObj);
        catch ME,
          warning('Error loading from jsonfile %s:\n%s',jsonfile,getReport(ME));
          uiwait(errordlg('Could not load from jsonfile (see console for error'));
          return;
        end
        obj.toTrack = toTrack;
        obj.nmovies = size(obj.toTrack.movfiles,1);
        obj.page = 1;
        obj.setNPages();
        obj.updateMovieList();
      else
        writeToTrackJSON(obj.toTrack,jsonfile);        
      end
      success = true;
    end

    function edit_movie_Callback(obj,h,e,itemi)
      moviei = obj.item2MovieIdx(itemi);
      set(h,'String',obj.toTrack.movfiles{moviei,1});
    end
    function edit_trk_Callback(obj,h,e,itemi)
      moviei = obj.item2MovieIdx(itemi);
      set(h,'String',obj.toTrack.trkfiles{moviei,1});
    end

  end
end

