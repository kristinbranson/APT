classdef TrackFPS
  methods (Static)
    function [t_predinnr, t_predothr, t_read_tot, t_overothr] = ...
        computeTimes(t_over, t_read, t_pred, t_pred_inner)
      % Inputs
      % t_over: (scalar) overall time 
      % t_read: (array of ngt) read times, per im. first typically way long
      % t_pred: (array of nbatch) pred times outer, per batch. first 
      %   typically way long
      % t_pred_inner: (array of nbatch) pred times inner, per batch. first
      %   typically way long.
      % 
      % Outputs
      % t_predinnr: total time spent on pred inner
      % t_predothr: total time spend on pred, but not inner
      % t_read_tot: total time spent on read
      % t_overothr: total remaining time spent
      
      t_read(1) = median(t_read);
      t_pred(1) = median(t_pred);
      t_pred_inner(1) = median(t_pred_inner);
      
      t_predinnr = sum(t_pred_inner);
      t_predothr = sum(t_pred-t_pred_inner);
      t_read_tot = sum(t_read);
      t_overothr = t_over - t_read_tot - t_predinnr - t_predothr;
      
      % pred other (pred overall, minus pred inner. first one here usually doesnt
      %  take extra long. still, replace first el with median just to be safe)
      % overallother = overall - read - predinner - predother
      %
      % Note, the replace-by-median procedure will cause any difference to fall
      % into the overall/other bucket
    end
    function t = getTimesTbl(bszs,s0)
      nbch = numel(bszs);
      s = struct(...
        'bsize',cell(0,1),...
        't_predinnr',cell(0,1),...
        't_predothr',[],...
        't_read_tot',[],...
        't_overothr',[]);
      for ib=1:nbch
        s(end+1,1).bsize = bszs(ib); %#ok<AGROW>
        [s(end).t_predinnr, s(end).t_predothr, s(end).t_read_tot, ...
          s(end).t_overothr] = TrackFPS.computeTimes(s0.t_overall(ib),...
          s0.t_read{ib},s0.t_pred{ib},s0.t_pred_inner{ib});
      end
      t = struct2table(s);
    end
  end
end