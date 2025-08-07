function [traj_mat, varargout] = interp_traj(beh_data, x_lim, y_lim, n, min_n, date, t_plot)
%
% loads in trajectory data and aligns all of the trajectories so that they
% go from S -> E. Important for doing SVD/PCA of trajectories
% 
% As of Mar. 14, 2022:
% The alignment procedure interpolates x and y position data (separately,
% as a function of sample number) to ensure all trajectories have the same
% number of datapoints. If the number of data points specified for
% interpolation is less than the number of data points in the trajectory,
% a roughly linear spacing of data points are selected for interpolation
%
% beh_data is a table containing variables: | trial | epoch | x | y |
% (can be created using video_process on raw camera data and
% position_process on the output of video_process)
%
% The x & y (maze) limits can be chosen, as well as the number of points
% to use in the trajectory (n). Specify t_plot as true to plot a subset of
% trials
%
% IMPORTANT: x and y are expected to be in scaled maze values, not pixel
% values! This means coordinates should be centered, rotated if needed, and
% scaled to the length of the maze before using this program!
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% center data
xpos = beh_data.x-mean(beh_data.x,"omitnan");
ypos = beh_data.y-mean(beh_data.y,"omitnan");
trials = unique(beh_data.trial(beh_data.trial>=1));
skips = [];
t_lens = [];
max_t = 0;

% for troubleshooting, plot some random trials
if t_plot
    rand_p = sort(randsample(trials,10));
else
    rand_p = [];
end


traj_mat = nan(2*n,max(trials)); % column of x, then y
for t = trials' 
    % filter for "choice" epoch
    tixs = beh_data.epoch=="choice" & beh_data.trial==t;
    if sum(tixs) < min_n
        disp(['Too few ixs (', num2str(sum(tixs)), ')'])
        disp(['Removing trial ',num2str(t)])
        skips = [skips;t];
        continue
    end
    xt = xpos(tixs); yt = ypos(tixs);

     % check for LV tracking screw-ups
     % HARDCODED - limits of maze (in cm)
    if abs(max(xt)) > 80 || abs(max(yt)) > 80
        skips = [skips;t];
        continue
    end

    % rotate trajectory so it goes S -> E
    if xt(end)<xt(1)
        xt = -xt;
    end
    if yt(end)<yt(1)
        yt = -yt;
    end
    % center again based on maze coords going S -> E
    x1 = xt-median(xt(yt<-10 & yt>y_lim),'omitnan');
    y1 = yt-median(yt(xt>(x_lim-10) & xt<x_lim),'omitnan');
    % get end (xix) and start (yix) indices
    xix = find(x1>=x_lim+2,1,'first'); % the +-2 gives a lil wiggle room
    yix = find(y1<=y_lim-2,1,'last');

    if isempty(yix)
        yix = 1;
    end
    if isempty(xix)
        xix = numel(x1);
    end

    x1 = x1(yix:xix); y1 = y1(yix:xix);
    % recenter the trimmed trajectories
    x1 = x1-median(x1(y1<-10 & y1>y_lim),'omitnan');
    y1 = y1-median(y1(x1>(20) & x1<x_lim),'omitnan');
    if isempty(x1) || isempty(y1) || xix<yix || sum(isnan(xt))>0
        disp([num2str(xix),' x ix, ', num2str(max(xt))])
        disp([num2str(yix),' y ix'])
        disp(['Removing trial ',num2str(t)])
        skips = [skips;t];
        continue
    end
    
    % interpolation wants unique values
    [~,xix,~] = unique(x1,'stable');
    [~,yix,~] = unique(y1,'stable');
    ixs = intersect(xix,yix);
    xt = x1(ixs); yt = y1(ixs);
    max_t = max([max_t,numel(x1)]);
    % check that we've still got a reasonable number of indices
    if numel(xt) < min_n
        disp("Removing trial "+t+" (not enough ixs)")
        numel(xt)
        skips = [skips;t];
        continue
    end

    try
        % drop nans
        xt = xt(~isnan(xt) & ~isnan(yt));
        yt = yt(~isnan(yt) & ~isnan(xt));
    catch
        % just some debugging info
        size(xt)
        size(yt)
        disp(t)
        skips = [skips;t];
        continue
    end
    
    % standardize trajectories to go from 0->1 in x,y
    xt = (xt-xt(1))/xt(end);
    yt = (yt-yt(1))/(abs(yt(end)-yt(1)));
    % make sure you have good number of position coords. for interpolation
    if numel(xt) <= n && numel(xt) >= min_n && numel(yt) >= min_n
        xt = interp1(1:numel(xt), xt, linspace(1,numel(xt),n), 'spline');
        yt = interp1(1:numel(yt), yt, linspace(1,numel(yt),n), 'spline');
    % if you have more than the number you want, linearly subsample
    elseif numel(xt) > n && numel(yt) > n
% % %         % randomly sample subset of trajectory indices, but make sure the
% % %         % subset doesn't skip huge portions of the data
% % %         max_skip = n/10; % indices can be no farther apart than this
% % %         nsamps = 0; % keep track of attempts
% % %         max_s = 10000;
% % % %         disp('resampling')
% % %         while max_skip>=n/10 && nsamps <= max_s
% % %             ixs = randsample(numel(xt),floor(3*n/4));
% % %             % keep original start and end indices
% % %             ixs(1) = 1; ixs(end) = numel(xt);
% % %             max_skip = max(diff(sort(ixs)));
% % %             nsamps = nsamps + 1;
% % %         end
% % %         if nsamps >= max_s
% % %             disp('Sampling may be uneven!')
% % %             disp(max_skip)
% % %         end
        ixs = unique(round(linspace(1,numel(xt),n)));
        % keep original start and end indices
        ixs(1) = 1; ixs(end) = numel(xt);
        xt = xt(sort(ixs)); yt = yt(sort(ixs));
        yt = interp1(1:numel(yt), yt, linspace(1,numel(yt),n), 'spline');
        xt = interp1(1:numel(xt), xt, linspace(1,numel(xt),n), 'spline');
%         disp('resampled')
    elseif numel(xt) < min_n
        disp(['Removing trial ',num2str(t)])
        disp(['Too few ixs (', num2str(sum(tixs)), ')'])
        skips = [skips;t];
        continue
    else
        figure
        scatter(xt, yt, 12, 'filled')
    end
    
    if isempty(xt) || isempty(yt)
        disp("trial "+t+" did not survive interpolation")
        skips = [skips;t];
        continue
    end
    
    % rat can't be in off-maze section
    % (this is now in norm'd coords, so should work broadly)
    if any(xt(yt<0.7)>0.6)
        disp(["Tracking off-maze, ",date, num2str(t)])
        figure
        plot(xt,yt)
        skips = [skips;t];
        continue
    end

    [X,Y] = meshgrid(-0.9:0.1:1);
    % change in x and y position
    Z = histcounts2(diff(xt),diff(yt),-1:0.1:1,-1:0.1:1);

    % use height of histogram and max dx and/or dy to exclude trajectories
    if max(Z(:)) <= 4 || max(abs(diff(xt)))>0.33 || max(abs(diff(yt)))>0.33 || xt(end)<xt(1) || abs(xt(end)-1)>0.3 || abs(yt(1)) > 0.3
        figure(99)
        subplot(1,3,3)
        pcolor(X,Y,Z)
        colorbar
        shading flat
        subplot(1,3,2)
        scatter(xt,yt,20,'filled')
        xlim([-0.5 1])
        subplot(1,3,1)
        scatter(x1,y1,20,'filled')
        title([date,', trial ',num2str(t)])
        hold off
        answer = questdlg('Skip this trial?', ...
        ['Trial ', num2str(t)], ...
        'Yes','No','Stop','No');
        % Handle response
        switch answer
            case 'Yes'
                disp('Okay, removing from data')
                skips = [skips;t];
                continue
            case 'No'
                disp('Great, moving on.')
            case 'Stop'
                disp('Whoops...')
                error('Shutting it down')
        end 
        close(99)
%         disp(['Removing trial ', num2str(t)])
%         skips = [skips;t];
    end
    
    if ismember(t,rand_p)
        figure
        scatter(xt,yt,20,'filled')
%         hold on
%         scatter(x1,y1,10, 'filled')
        title(['trial ',num2str(t)])
    end
    
    
    try
        traj_mat(:,t) = [xt, yt];
        
    catch
        size(xt)
        size(yt)
        traj_mat(:,t) = [xt; yt];
        t
    end
    t_lens = [t_lens;sum(tixs)];

end

% skip these for now so that size(traj_mat,2) == size(coh_mat,2)
% traj_mat = traj_mat(:,~ismember(trials,skips));
% traj_mat = traj_mat(:, sum(isnan(traj_mat)) == 0);

if nargout == 2
    varargout{1} = skips;
elseif nargout == 3
    varargout{1} = skips;
    varargout{2} = max_t;
end 

end