classdef APT_helper
  %helper functions for APT.
  % mostly to enable to quick scripting
  
  methods (Static)
    function lObj = load_proj(lblfile,varargin)
      % load project and if required update paths 
      APT.setpathsmart();
      [replace_path] = myparse(varargin,'replace_path',{'',''});
      lObj = Labeler('projfile',lblfile,'replace_path',replace_path);      
    end
    

    % quick_train() seems messed up, b/c it refers to a variable "self", which is
    % never set, and this is static method.  So commenting out for now.  
    % -- ALT, 2023-05-14
%     function quick_train(lObj,varargin)
%           [net_type,backend,niters,test_tracking,block,serial2stgtrain, ...
%         batch_size, params, aws_params] = myparse(varargin,...
%             'net_type','grone','backend','docker',...
%             'niters',1000,'test_tracking',true,'block',true,...
%             'serial2stgtrain',true,...
%             'batch_size',8,...
%             'params',[],... % optional, struct; see structsetleaf
%             'aws_params',struct());
%           
%       if ~isempty(net_type)
%         self.setup_alg(net_type)
%       end
%       self.set_params_base(self.info.has_trx,niters,self.info.sz, batch_size);
%       if ~isempty(params)
%         sPrm = self.lObj.trackGetParams();
%         sPrm = structsetleaf(sPrm,params,'verbose',true);
%         self.lObj.trackSetParams(sPrm);
%       end
%       self.set_backend(backend,aws_params);
% 
%       lObj = self.lObj;
%       handles = lObj.gdata;
%       %oc1 = onCleanup(@()ClearStatus(handles));
%       wbObj = WaitBarWithCancel('Training');
%       oc2 = onCleanup(@()delete(wbObj));
%       centerOnParentFigure(wbObj.hWB,handles.figure);
%       tObj = lObj.tracker;
%       tObj.skip_dlgs = true;
%       if lObj.trackerIsTwoStage && ~strcmp(backend,'bsub')
%         tObj.forceSerial = serial2stgtrain;
%       end      
%       lObj.trackRetrain('retrainArgs',{'wbObj',wbObj});
%       if wbObj.isCancel
%         msg = wbObj.cancelMessage('Training canceled');
%         msgbox(msg,'Train');
%       end      
% 
%     end
    
    
%     function check_track(lObj,varargin)
%       [backend] = myparse(varargin,'backend','docker');      
%     end
    
    function load_trk(trkfiles)
      lObj = APT_helper.find_lObj();
      mov = lObj.currMovie;      
      lObj.labels2ImportTrk(mov,trkfiles);
    end
    
    function load_label_table()
      
    end
    
%     function set_backend(lObj, backend)
%       beType = DLBackEndFromString(backend) ;
%       be = DLBackEndClass(beType,lObj.trackGetDLBackend);
%       lObj.trackSetDLBackend(be);
%     end
    
    function lObj = new_proj(proj_name, npts,ismulti,varargin)
      [nviews,has_trx] = myparse(varargin,...
        'nviews',1,'has_trx',false);
      cfg = ReadYaml(Labeler.DEFAULT_CFG_FILENAME);
      cfg.NumViews = nviews;
      cfg.NumLabelPoints = npts;
      cfg.Trx.HasTrx = has_trx;
      cfg.MultiAnimal = ismulti;
      cfg.ViewNames = {};
      cfg.LabelPointNames = {};
      cfg.Track.Enable = true;
      cfg.ProjectName = proj_name;
      FIELDS2DOUBLIFY = {'Gamma' 'FigurePos' 'AxisLim' 'InvertMovie' 'AxFontSize' 'ShowAxTicks' 'ShowGrid'};
      cfg.View = repmat(cfg.View,cfg.NumViews,1); 
      for i=1:numel(cfg.View)
        cfg.View(i) = ProjectSetup('structLeavesStr2Double',cfg.View(i),FIELDS2DOUBLIFY);
      end

      lObj = Labeler;
      lObj.initFromConfig(cfg);
      lObj.projNew(cfg.ProjectName);
      lObj.notify('projLoaded');

    end
        
    function set_track_label_sz(lObj)
      [font_sz, font_color] = myparse('font_sz',25,'font_color',[0 0 1]);
      set(lObj.labeledpos2trkViz.tvtrx.hTrxTxt,'FontSize',font_sz,'Color',font_color);
    end
    
    function lObj = find_lObj()
      h = findall(0,'type','figure');
      for ndx = 1:numel(h)
        if isfield(guidata(h(ndx)),'labelerObj')
          ll = guidata(h(ndx));
          lObj = ll.labelerObj;
        end
          
      end
    end

    function enlarge_id()
      lobj = APT_helper.find_lObj();
      set(lobj.labeledpos2trkViz.tvtrx.hTrxTxt,'FontSize',25,'Color',[0 0 1]);
    end
    
  end
  
end