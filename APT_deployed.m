function APT_deployed(do_test,test_name,net_type)

try
  
  if nargin < 1
    do_test=false;
  else
    do_test = str2double(do_test) > 0.5;
  end
  
  if nargin<2
    test_name = 'alice';
  end
    
  if nargin<3
    if strcmp(test_name,'roianma2')
      net_type = 1;
    else
      net_type = 'mdn_joint_fpn';
    end    
  end
  
  if do_test
    tobj = TestAPT('name',test_name);
    tobj.test_setup('simpleprojload',1)
    params = {};
    tobj.test_train('net_type',net_type,'params',params,'niters',100);
  else
    StartAPT()
  end
catch ME
  errordlg(getReport(ME,'extended','hyperlinks','off'))  
end
