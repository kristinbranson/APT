classdef DeepTrackerTrainingMonitor < handle
  properties
    hfig % scalar fig
    haxs % [2] axis handle, viz training loss, dist
    hline % [nviewx2] line handle, one loss curve per view
    
    resLast % last training json contents received
  end
  
  methods
    function obj = DeepTrackerTrainingMonitor(nview)
      obj.hfig = figure('Visible','on');
      obj.haxs = reshape(createsubplots(2,1,.1),2,1);
      arrayfun(@(x)grid(x,'on'),obj.haxs);
      arrayfun(@(x)hold(x,'on'),obj.haxs);
      title(obj.haxs(1),'Training Monitor','fontweight','bold');
      xlabel(obj.haxs(2),'Iteration');
      ylabel(obj.haxs(1),'Loss');
      ylabel(obj.haxs(2),'Dist');
      linkaxes(obj.haxs,'x');
      
      clrs = lines(nview);
      h = gobjects(nview,2);
      for ivw=1:nview
        for j=1:2
          h(ivw,j) = plot(obj.haxs(j),nan,nan,'.-','color',clrs(ivw,:));
        end
      end
      obj.hline = h;
      obj.resLast = [];
    end
    function delete(obj)
      deleteValidHandles(obj.hfig);
      obj.hfig = [];
      obj.haxs = [];
    end
    function resultsReceived(obj,sRes)
      % Callback executed when new result received from training monitor BG
      % worker
      
      res = sRes.result;
      tfAnyUpdate = false;
      
      h = obj.hline;
      for ivw=1:numel(res)
        if res(ivw).jsonPresent && res(ivw).tfUpdate
          contents = res(ivw).contents;
          set(h(ivw,1),'XData',contents.step,'YData',contents.train_loss);
          set(h(ivw,2),'XData',contents.step,'YData',contents.train_dist);
          lclAutoAxisWithYLim0(obj.haxs(1));
          lclAutoAxisWithYLim0(obj.haxs(2));
          tfAnyUpdate = true;
        end
      end
      
      if isempty(obj.resLast) || tfAnyUpdate
        obj.resLast = res;
      end

%           
%           
%         fprintf(1,'View%d: jsonPresent: %d. ',ivw,res(ivw).jsonPresent);
%         if res(ivw).tfUpdate
%           fprintf(1,'New training iter: %d.\n',res(ivw).lastTrnIter);
%         elseif res(ivw).jsonPresent
%           fprintf(1,'No update, still on iter %d.\n',res(ivw).lastTrnIter);
%         else
%           fprintf(1,'\n');
%         end
    end
  end
  
end

function lclAutoAxisWithYLim0(ax)
axis(ax,'auto');
yl = ylim(ax);
yl(1) = 0;
ylim(ax,'yl');
end
