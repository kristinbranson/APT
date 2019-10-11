function flynum = getflynum(inputString)
%takes project filename as input outputs fly number
%
%only tested for tiny subset - check works before using!!!!

    [p,fname,ext]=fileparts(inputString);
    
        
    flyi = strfind(fname,'fly_');   
    if ~isempty(flyi)    
        try
            flynum = str2num(fname(flyi+4:flyi+7));
        catch
            flynum=[];
        end
    else
        flyi = strfind(fname,'fly');
        try
            flynum = str2num(fname(flyi+3:flyi+6));  
        catch
            flynum=[];
        end     
    end

    if isempty(flynum)
        
        flyi = strfind(fname,'fly_');   
        if ~isempty(flyi)    
            flynum = str2num(fname(flyi+4:flyi+6));
        else
            flyi = strfind(fname,'fly');
            flynum = str2num(fname(flyi+3:flyi+5));       
        end
            
    end
    
    
   if isempty(flynum)
       fname
       flyi
       flynum
       error('getflynum.m failed')
   end