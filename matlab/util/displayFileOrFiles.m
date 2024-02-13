function displayFileOrFiles(fileNameOrFileNames, bgWorkerObj)    
    if iscell(fileNameOrFileNames) ,
        fileNames = fileNameOrFileNames ;
        for j = 1 : length(fileNames) ,
            fileName = fileNames{j} ;
            fprintf(1,'\n### %s\n\n',fileName);
            fileContents = bgWorkerObj.fileContents(fileName);
            disp(fileContents);
        end
    else
        % Should be a char array or string
        fileName = fileNameOrFileNames ;
        fprintf(1,'\n### %s\n\n',fileName);
        fileContents = bgWorkerObj.fileContents(fileName);
        disp(fileContents);
    end
end
