function flynum = getflynum(inputString)
%takes project filename as input outputs fly number
%
%only tested for tiny subset - check works before using!!!!

    [p,fname,ext]=fileparts(inputString);
    
        
    flyi = strfind(fname,'fly_');
    flynum = str2num(fname(flyi+4:flyi+6));
    
    if isempty(flyi)
        
        flyi = strfind(fname,'fly');
        flynum = str2num(fname(flyi+3:flyi+5));
        
    end

   if isempty(flynum)
       fname
       flyi
       flynum
       error('getflynum.m failed')
   end