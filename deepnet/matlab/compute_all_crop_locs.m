function compute_all_crop_locs()
	lbl_file = '/groups/huston/hustonlab/flp-chrimson_experiments/fly2BodyAxis_lookupTable_Ben.csv';
%	lbl_file = '/groups/branson/bransonlab/mayank/stephen_copy/fly2BodyAxis_lookupTable_Ben.csv';
	r_params = '/groups/branson/bransonlab/mayank/stephen_copy/crop_regression_params.mat';
	crop_size = {[230,350],[350,350]};
	R = load(r_params);
	fid = fopen(lbl_file,'r');
	C = textscan(fid,'%d,%s');
	fclose(fid);
	all_crops = cell(size(C{1},1),3);
	for ndx = 1:size(C{1},1)
		all_crops{ndx,1} = C{1}(ndx);
		if ~exist(C{2}{ndx},'file'),
			continue;
                end
                lblfile = C{2}{ndx};
 		L = loadLbl(lblfile);      
		if class(L.labeledpos{1}) == 'double'
			curpts = L.labeledpos{1}(:,:,1);
		else
			pts = SparseLabelArray.full(L.labeledpos{1});
			curpts = pts(:,:,1);
		end
		for view  = 1:2
			hh = L.movieInfoAll{1,view}.info.Height;
			ww = L.movieInfoAll{1,view}.info.Width;
			neck_locs = squeeze(curpts(6 + 10*(view	-1),:));
			x_str = sprintf('reg_view%d_x',view);
			y_str = sprintf('reg_view%d_y',view);
			x_reg = R.(x_str);
			y_reg = R.(y_str);
			x_left = round(x_reg(1) + x_reg(2)*neck_locs(1));
			if x_left < 1
				x_left = 1;
			end
			x_right = x_left +  crop_size{view}(1)-1;
			if x_right > ww
				x_left = ww - crop_size{view}(1) + 1;
				x_right = ww;
			end	
			y_top = round(y_reg(1) + y_reg(2)*neck_locs(2));
			if y_top < 1
				y_top = 1;
			end
			y_bottom = y_top +  crop_size{view}(2)-1;
			if y_bottom > hh
				y_top = hh - crop_size{view}(2) + 1;
				y_bottom = hh;
			end	
			all_crops{ndx,view+1} = [x_left,x_right,y_top,y_bottom];


		end
		fprintf('%d\n',ndx);
	end

	save('/groups/branson/bransonlab/mayank/stephen_copy/crop_locs_all.mat','all_crops')
