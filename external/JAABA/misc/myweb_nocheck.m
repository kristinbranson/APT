function [stat,msg] = myweb_nocheck(html_file)

if ismac,
  [stat,msg] = unix(['open ' html_file]);
elseif isunix,
  browsers = {'xdg-open','google-chrome','firefox','opera','konqueror','epiphany'};
  browser = '';
  for i = 1:numel(browsers),
    [a,b] = unix(sprintf('which %s',browsers{i}));
    b = strtrim(b);
    if a == 0 && ~isempty(b),
      browser = b;
    end
  end
  if isempty(browser),
    stat = 1;
    msg = sprintf('Could not find a web browser');
    return
  end
  [stat,msg] = unix(sprintf('%s %s',browser,html_file));  
elseif ispc,
  html_file = fullfile(html_file);
  [stat,msg] = dos(['cmd.exe /c rundll32 url.dll,FileProtocolHandler "' html_file '"']);
end
