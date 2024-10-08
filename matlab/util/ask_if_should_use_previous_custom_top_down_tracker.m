function use = ask_if_should_use_previous_custom_top_down_tracker(prev_tracker)

use = false;
if ~isempty(prev_tracker) && prev_tracker.valid
  prev_net_str = ...
    sprintf('%s (stage 1) and %s (stage 2)',...
            prev_tracker.stage1Tracker.trnNetType.displayString, ...
            prev_tracker.trnNetType.displayString );
  qstr = ...
    sprintf(strcatg('Continue with previous custom tracker with %s?. ', ...
                    'Note that if you change, you will lose any previously trained tracker of this type.'),...
            prev_net_str );
  res = questdlg(qstr,....
                 'Change custom two-stage tracker?',...
                 'Continue','Change','Continue');
  if strcmp(res,'Continue')
    use = true;
    return
  end
end

end  % function

