function TrkInfoUI(lobj)

assert(lobj.maIsMA,'UI is functional only for multi-animal projects');
f = findall(groot(),'tag','TrkInfoUI');
if isempty(f)
  init_figure(lobj);
else
  figure(f);
  h = guidata(f);
  if h.curmov ~= lobj.currMovie
    curmov = lobj.currMovie;
    h.mov_tbl.Selection = [curmov 1];
    h.curmov = curmov;
    h.lobj = lobj;
    guidata(f,h);
    update_movie(f);
  end
end
end

function f = init_figure(lobj)

f = uifigure('Units','pixel','Position',[250,250,1000,800],...
             'tag','TrkInfoUI','Name','Track Info');
%     centerOnParentFigure(f,lobj.hFig)
h = guidata(f);
h.lobj = lobj;
h.fig = f;

curmov = lobj.currMovie;
nr = 3;
nc = 2;
gl = uigridlayout(f,[nr nc]);

gl.RowHeight = {150,'1x',50};
gl.ColumnWidth = {'1x',100};

sel_btn = uiswitch(gl,'slider','ValueChangedFcn',@pred_imported_callback);
sel_btn.Layout.Row = 1;
sel_btn.Layout.Column = 2;
sel_btn.Items = {'Imported','Predicted'};
sel_btn.Value = {'Imported'};
sel_btn.Orientation = 'Vertical';
h.sel_btn = sel_btn;


mov_list = lobj.movieFilesAllFullGTaware;
mov_tbl = uitable(gl,'Data',mov_list);
mov_tbl.Layout.Row = 1;
mov_tbl.Layout.Column = 1;
mov_tbl.ColumnName = {'Movie'};
mov_tbl.ColumnWidth = {'1x'};
mov_tbl.CellSelectionCallback = {@cell_click_mov,f};
h.mov_tbl = mov_tbl;

tbl = uitable(gl);
tbl.Layout.Row = 2;
tbl.Layout.Column = [1 nc];
tbl.CellSelectionCallback = {@cell_click_tbl,f};
h.tbl = tbl;

gl_btm = uigridlayout(gl,[1 4]);
gl_btm.Layout.Row = 3;
gl_btm.Layout.Column = [1 nc];


sf_btn = uibutton(gl_btm,'Text','First Frame','ButtonPushedFcn',@sf_btn_callback);
sf_btn.Layout.Row = 1;
sf_btn.Layout.Column = 1;
h.sf_btn = sf_btn;
ef_btn = uibutton(gl_btm,'Text','End Frame','ButtonPushedFcn',@ef_btn_callback);
ef_btn.Layout.Row = 1;
ef_btn.Layout.Column = 2;
h.ef_btn = ef_btn;

prev_btn = uibutton(gl_btm,'Text','Prev break','ButtonPushedFcn',@prev_btn_callback);
prev_btn.Layout.Row = 1;
prev_btn.Layout.Column = 3;
h.prev_btn = prev_btn;
next_btn = uibutton(gl_btm,'Text','Next break','ButtonPushedFcn',@next_btn_callback);
next_btn.Layout.Row = 1;
next_btn.Layout.Column = 4;
h.next_btn = next_btn;


h.sf = [];
h.ef = [];
h.breaks = {};
h.top_links = {};
h.data = {};
h.has_data = false;
h.curtrk = [];
mov_tbl.Selection = [curmov 1];
h.curmov = curmov;

guidata(f,h);
update_movie(f);

end

function cell_click_mov(table_handle, event, fig_handle)
h = guidata(fig_handle);
table_handle.Selection = [h.curmov 1];
pause(0.5); % to allow the user time to add a second click
if strcmpi(fig_handle.SelectionType, 'open')
  h.curmov = event.Indices(1);
  table_handle.Selection = [h.curmov 1];
  guidata(fig_handle,h);
  update_movie(fig_handle);
end
end

function cell_click_tbl(table_handle, event, fig_handle)
pause(0.5); % to allow the user time to add a second click
if strcmpi(fig_handle.SelectionType, 'open')
  switch_target(guidata(fig_handle),event.Indices(1));
end
end

function update_movie(f)
h = guidata(f);
idx = h.curmov;
lobj = h.lobj;
imp_trk = lobj.labels2GTaware{idx};   
if ~isempty(lobj.tracker)
  pred_trk = lobj.tracker.getTrackingResults(MovieIndex(idx));
  pred_trk = pred_trk{1};
else
  pred_trk = [];
end

has_imp = imp_trk.hasdata;
has_pred = ~isempty(pred_trk) && pred_trk.hasdata;

if has_imp && has_pred
  h.sel_btn.Enable = 'on';
elseif has_imp
  h.sel_btn.Enable = 'off';
  h.sel_btn.Value = 'Imported';
elseif has_pred
  h.sel_btn.Enable = 'off';
  h.sel_btn.Value = 'Predicted';
else
  h.sel_btn.Enable = 'off';
  h.tbl.Data = {};
  h.trk = {};
  h.has_data = false;
  return;
end

if strcmp(h.sel_btn.Value,'Imported')
  trk = imp_trk;
else
  trk = pred_trk;
end

[dat,sf,ef,breaks,top_links] = get_data(trk);
h.tbl.Data = dat;
h.sf = sf;
h.ef = ef;
h.breaks = breaks;
h.top_links = top_links;
h.data = dat;
h.tbl.ColumnSortable = true;
h.trk = trk;
h.has_data = true;

guidata(f,h);
end
  

function [tdat,sf,ef,breaks,top_links] = get_data(trk)
  n_top = 50;
  n_trk = trk.ntracklets;
  sf = trk.getStartFrame;
  ef = trk.getEndFrame;
  varNames = {'ID','N Frm','N Frm Tot', 'Start','End',...
    'Brks','Avg Bout Sz','Avg Brk Sz',...
    'Median Link','Max Link','90 Prc Link'};
  nvar = numel(varNames);
  tdat = table('Size',[n_trk,nvar],'VariableTypes',repmat({'double'},[1,nvar]),...
    'VariableNames',varNames);
  breaks = cell(1,n_trk);
  top_links = [];
  %%
  
  for ndx = 1:n_trk
    nfr = ef(ndx)-sf(ndx)+1;
    curt = trk.getPTrkTgt(ndx);
    valid_pred = shiftdim(~all(isnan(curt(:,1,:)),1),2);
    n_pred = nnz(valid_pred);
    [si,ei] = get_interval_ends(valid_pred);
    n_breaks = numel(si)-1;
    int_size = mean(ei-si);
    if n_breaks>0
      break_size = si(2:end) - ei(1:end-1);
      avg_break_size = mean(break_size);
    else
      avg_break_size = 0;
    end
    
    link = abs(curt(:,:,2:end)-curt(:,:,1:end-1));
    link = nanmean(sum(link,2),1);
    link = shiftdim(link,2);
    link = link(~isnan(link));
    med_link = nanmedian(link);
    if numel(link)>0
      max_link = nanmax(link);
    else
      max_link = NaN;
    end
    link_90 = prctile(link,90);
    id = trk.pTrkiTgt(ndx);
    tdat(ndx,:) = {id,n_pred, nfr, sf(ndx), ef(ndx), n_breaks, ...
      int_size, avg_break_size,med_link,max_link,link_90};
    breaks{ndx} = [si,ei];
  end
  
end

function pred_imported_callback(handle,event,~)
 h = guidata(handle);
 h.sel_btn.Value = event.Value;
 guidata(handle,h)
 update_movie(h.fig);
 
end

function h = switch_target(h,tgt)
if ~h.has_data, return; end
if h.lobj.currMovie ~= h.curmov
  qstr = sprintf('Switch to movie %d?',h.curmov);
  res = questdlg(qstr,'Switch Movie');
  if strcmp(res, 'Yes')
    h.lobj.movieSetGUI(h.curmov);
  else
    return
  end
end
lobj = h.lobj;
sf = h.sf;
ef = h.ef;
trk = h.trk;
curfr = lobj.currFrame;
if (sf(tgt)>curfr) || (ef(tgt)<curfr)
  lobj.setFrameGUI(sf(tgt));
else
  haspred = trk.getPTrkFT(curfr,tgt);
  if ~haspred
    % set frame to the closest frame that has prediction for the current
    % fly
    tdat = trk.getPtrkTgt(tgt);
    vfr = find(~all(isnan(tdat),[1,2]));
    closest = argmin(abs(vfr-curfr+sf));
    lobj.setFrameGUI(vfr(closest));
  end
end

if strcmp(h.sel_btn.Value, 'Predicted')
  lobj.tracker.trkVizer.trxSelectedTrxID(tgt,true);
  lobj.tracker.trkVizer.centerPrimary;
else
  lobj.labeledpos2trkViz.trxSelectedTrxID(tgt,true);
  lobj.labeledpos2trkViz.centerPrimary;
end
h.curtrk = tgt;
guidata(h.fig,h);
end

function centerPrimary(h)
lobj = h.lobj;
if strcmp(h.sel_btn.Value, 'Predicted')
  lobj.tracker.trkVizer.centerPrimary;
else
  lobj.labeledpos2trkViz.centerPrimary;
end

end
function prev_btn_callback(handles,event,~)
h = guidata(handles);
if isempty(h.tbl.Selection)
  return;
end
curtrk = h.tbl.Selection(1);
trk = h.trk;
lobj = h.lobj;

sf = trk.startframes(curtrk);
ef = trk.endframes(curtrk);
valid = trk.getPTrkFT(sf:ef,curtrk);
[ss,ee] = get_interval_ends(valid);
ee = ee-1;
ss = [ss;ee];
curfr = lobj.currFrame;
if curfr<sf
  warning('No previous breaks');
elseif curfr>ef
  lobj.setFrameGUI(ef);
else
  ss(ss>=(curfr-sf+1)) = nan;
  if all(isnan(ss))
    lobj.setFrameGUI(sf);
  else
    sndx = argmax(ss);
    lobj.setFrameGUI(ss(sndx)+sf-1);
  end
end
if isempty(h.curtrk) || (h.curtrk ~= curtrk)
  h = switch_target(h,curtrk);
end
centerPrimary(h);
guidata(handles,h);
end

function next_btn_callback(handles,event,~)
h = guidata(handles);
if isempty(h.tbl.Selection)
  return;
end
curtrk = h.tbl.Selection(1);
trk = h.trk;
lobj = h.lobj;

sf = trk.startframes(curtrk);
ef = trk.endframes(curtrk);
valid = trk.getPTrkFT(sf:ef,curtrk);
[ss,ee] = get_interval_ends(valid);
ee = ee-1;
ss = [ss;ee];
curfr = lobj.currFrame;
if curfr>ef
  warning('No next breaks');
elseif curfr<sf
  lobj.setFrameGUI(sf);
else
  ss(ss<=(curfr-sf+1)) = nan;
  if all(isnan(ss))
    lobj.setFrameGUI(ef);
  else
    sndx = argmin(ss);
    lobj.setFrameGUI(ss(sndx)+sf-1);
  end
end
if isempty(h.curtrk) || (h.curtrk ~= curtrk)
  h = switch_target(h,curtrk);
end
centerPrimary(h);
guidata(handles,h);
end

function sf_btn_callback(handles,event,~)
h = guidata(handles);
if isempty(h.tbl.Selection)
  return;
end
curtrk = h.tbl.Selection(1);
trk = h.trk;
lobj = h.lobj;

sf = trk.startframes(curtrk);
lobj.setFrameGUI(sf);
if isempty(h.curtrk) || (h.curtrk ~= curtrk)
  h = switch_target(h,curtrk);
end
centerPrimary(h);
guidata(handles,h);
end

function ef_btn_callback(handles,event,~)
h = guidata(handles);
if isempty(h.tbl.Selection)
  return;
end
curtrk = h.tbl.Selection(1);
trk = h.trk;
lobj = h.lobj;

ef = trk.endframes(curtrk);
lobj.setFrameGUI(ef);
if isempty(h.curtrk) || (h.curtrk ~= curtrk)
  h = switch_target(h,curtrk);
end
centerPrimary(h);
guidata(handles,h);
end

