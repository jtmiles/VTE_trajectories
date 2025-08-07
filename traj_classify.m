% set some trajectory params
n = 200;
x_lim = 40; % maze coords in cm
y_lim = -25; % maze coords in cm
min_n = 10; %floor(n/16);

% traj_mat has x pos stacked on top of y pos
traj_mat = [];
all_data = [];
pdir = uigetdir(cd,"Select position data director"); 
cd(pdir)
tf_info = dir('*.csv');
tfiles = {tf_info.name};
tdir = uigetdir(cd,"Select task data directory");
cd(tdir)
bf_info = dir('*behav.txt');
bfiles = {bf_info.name};

%%%% A lot of this file loading is ensuring I had correct alignment between
%%%% behavior trial metadata and trajectories per trial and will not be
%%%% useful for anyone using different data structures. I've tried to
%%%% denote import stuff with %! either inline or above and below a block
%%%% of code that is actually useful

bdata = [];
zlnidphi = [];
trajfits = [];

assert(numel(bfiles) == numel(tfiles),'Different number of position and task files')
bcols = {'trial','block','trial_in_block','reward_arm','outcome',...
         'start_arm','chosen_arm','delay'};

for ft = tfiles
    tfile = ft{1};
    [d_start, d_end] = regexp(tfile, '(\d{4})-(\d{2})-(\d{2})');
    date = tfile(d_start:d_end);
    r_id = tfile(d_end+2:d_end+4);
    
    for fb = bfiles
        if ~isempty(strfind(fb{1},date)) && ~isempty(strfind(fb{1},r_id))
           cd(pdir)
           traj_data = readtable(tfile);
           traj_data(string(traj_data.epoch)=="post-session",:) = [];
           cd(tdir)
           beh_data = readtable(fb{1});
           beh_data.Properties.VariableNames = bcols;
           beh_ts = unique(beh_data.trial(beh_data.trial>=1));
           traj_data = traj_data(ismember(traj_data.trial,beh_ts),:);
           break
        else
            continue
        end %if
    end %for
    
    %! 
    % necessary function! apply to each file with trajectory data
    [trajs, skips, ~] = interp_traj(traj_data, x_lim, y_lim, n, min_n, date, false);
    %!

    disp([r_id,', ' date, ' ',num2str(size(trajs,2)), ' trials'])
    all_data = [all_data,trajs];

    beh_data = readtable(fb{1});
    beh_data.Properties.VariableNames = bcols;
    beh_data = beh_data(~ismember(beh_ts,skips),:);
    beh_ts = unique(beh_data.trial);

    trajts = unique(traj_data.trial(traj_data.trial>=1 & ~ismember(traj_data.trial,skips)));
    traj_mat = [traj_mat,trajs(:,beh_ts(~ismember(beh_ts,skips)))];

    if ~all(trajts==beh_ts)
        disp("trial mismatch "+string(r_id)+" "+string(date))
        mismatch = trajts~=beh_ts;
        if unique(trajts(mismatch)-beh_ts(mismatch)) == 1
            trajts(mismatch) = trajts(mismatch)-1;
        elseif unique(trajts(mismatch)-beh_ts(mismatch)) == -1
            trajts(mismatch) = trajts(mismatch)+1;
        end
    end
    
    %!
    % calculate zlnidphi (Redish lab measurement for VTEs)
    % useful to sanity check with another measure or to apply as filter
    % after clustering during cluster reassignment
    traj_data = traj_data((traj_data.trial>=1 & ~ismember(traj_data.trial,skips)),:);
    tgrp = findgroups(traj_data.trial);
    zlnidphi = [zlnidphi;...
                normalize(splitapply(@calc_idphi,table2array(traj_data(:,["x","y"])),tgrp))];
    assert(~any(isnan(zlnidphi)),"introducing nans into data")
    
    %!
    % R^2 for polynomial fit of trajectories (from Miles et al. 2021)
    % (same as for zlnidphi)
    fitvec = nan(numel(unique(tgrp)),1);
    for t = unique(tgrp)'
        [p,polyS] = polyfit(traj_data.x(tgrp==t),traj_data.y(tgrp==t),3);
        fitvec(t) = 1 - (polyS.normr/norm(traj_data.y(tgrp==t) - mean(traj_data.y(tgrp==t))))^2;
    end
    trajfits = [trajfits; fitvec];
    %!

    rcol = table(repmat(str2double(r_id),size(beh_data,1),1),'VariableNames',{'rat_ID'});
    dcol = table(repmat(date,size(beh_data,1),1),'VariableNames',{'date'});
    beh_data = [beh_data,dcol,rcol];
    bdata = [bdata;beh_data];
    clear beh_data skips;
    
    assert(numel(trajfits)==size(traj_mat,2),'Matrices are mismatched!')
    
end

try
    close(99)
catch
    
end

size(traj_mat)

%%%% most actually useful code starts here
%% Plot re-aligned trajectories

figure
plot(traj_mat(1:n,:), traj_mat(n+1:end,:), 'Color', [0 0 0 0.1], ...
    'LineWidth',2)
axis off

color_mat = [0.821	0.394	0.174
             0.184	0.392	0.580
             0.497  0.502   0.541
             0.523	0.071	0.222
             0.217	0.595	0.339
             0.905  0.823   0.111
             0.459	0.231	0.184
             0.09	0.325	0.851
             0.161	0.208	0.137];

%% Run SVD on the data
max_dim = 15; % PCs/SVs to keep
ztraj = normalize(traj_mat,2,"zscore","std");

ztraj([1 n+1 2*n],:) = 0;
% scaling and centering makes it PCA (I'm pretty sure?)
[zU,zS,zV] = svd(ztraj);
% this program creates the projection space w/ z-scored trajectories
% technically SVD doesn't need scaling/centering, and the coordinates
% themselves are pretty well constrained already, so choose your fighter!
[U,S,V] = svd(traj_mat./sqrt(size(traj_mat,2)),'econ');

% plotting both SVD and PCA just for comparison
figure
subplot(1,2,1)
scatter(1:numel(diag(S)),log(diag(S)),'filled')
hold on
scatter(1:numel(diag(zS)),log(diag(zS)),'filled')
xlim([0 max_dim])
title(['Scree plot (first ' num2str(max_dim),' PCs)'])

cuvar = cumsum(diag(zS).^2./sum(diag(zS).^2));
subplot(1,2,2)
scatter(1:numel(diag(S)),cumsum(diag(S).^2./sum(diag(S).^2)),12,'filled')
hold on
plot(1:numel(diag(zS)),cuvar, 'LineWidth',3)
xlim([0 max_dim])
ylim([0 1.05])
title('Cumul. var. explained')

%%
% to change percent cumulative variance explained
keep_d = find(cuvar<=0.95,1,'last');

figure
hold on
plot(1:numel(diag(zS)),cuvar, 'LineWidth',3)
line([1 1]* keep_d,[0 cuvar(keep_d)],'Linestyle',':','Color','k')
line([0 keep_d],[cuvar(keep_d) cuvar(keep_d)],'Linestyle',':','Color','k')
xlim([1 15])
ylim([0 1.05])
yticks(0:0.2:1)
ylabel('Cumul. var. explained')
xlabel('Number of PCs')
hold off

figure
for d = 1:keep_d
    plot(U(1:n,d),U(n+1:end,d),'LineWidth',2)
    hold on
end
hold off

%% create dendrogram on PC projection
dims = 1:keep_d;
proj_space = zU'*ztraj;
% proj_space = zS*zV';
proj_space = [proj_space(dims,:)];
Z = linkage(proj_space','ward');
figure
[~,~,dendp] = dendrogram(Z,size(Z,1));
xticks([])

%% cut dendrogram to determine clusters

% there are automated ways of returning n-number of clusters based on
% height at which you cut the dendrogram or silhouette values or cophenetic
% correlation coefficients, but I ended up just looking at the dendrogram 
% and picked where to cut based on how well it seemed to aggregate data
%
% in our work, there's really only VTE or non-VTE, so cutting to give two 
% clusters generally worked well.
%
% during development, though, I saw some instances where it looked like
% asking for more clusters and merging similar clusters that for whatever
% reason did not "agglomerate" in the way I'd expect was helpful. worth
% just messing with this and clicking through the cluster clouds to see
% what you get (code for that is below)

for d = [140 160]
    hclus = cluster(Z,'cutoff',d,'criterion','distance');
    if max(hclus) > size(color_mat,1)
        continue
    end

    figure;
    subplot(1,3,1)
    scatter(proj_space(1,:),proj_space(2,:),20,...
             color_mat(hclus,:),'filled',...
             'MarkerFaceAlpha',0.5)
    xlabel('PC1')
    ylabel('PC2')

    subplot(1,3,2)
    scatter(proj_space(2,:),proj_space(3,:),20,...
             color_mat(hclus,:),'filled',...
             'MarkerFaceAlpha',0.5)
    xlabel('PC2')
    ylabel('PC3')

    subplot(1,3,3)
    scatter(proj_space(3,:),proj_space(4,:),20,...
             color_mat(hclus,:),'filled',...
             'MarkerFaceAlpha',0.5)
    xlabel('PC3')
    ylabel('PC4')


    avg_clus = [];
    
    svals = silhouette(proj_space(dims,:)',hclus,'Euclidean');
    disp([d,max(hclus)])
    disp(sum(svals))
%     figure, histogram(svals,'BinEdges',-1:0.1:1,'Normalization','cdf')
    figure
    for c = 1:max(hclus)
        subplot(1,max(hclus),c)
        avg_clus(:,c) = median(traj_mat(:,hclus==c),2);
        plot(traj_mat(1:n,hclus==c),traj_mat(n+1:end,hclus==c),...
             'Color', [color_mat(c,:), 0.1],'LineWidth',2)
        hold on
        plot(avg_clus(1:n,c),avg_clus(n+1:end,c),'LineWidth',2,'Color',[0.3 0.3 0.3 0.7])
        xlim([-1 1.1])
        ylim([0 1.5])
        text(0.3,0.4,['n = ', num2str(sum(hclus==c))])
        title('Original clustering')
    end

end

%% Fine tuning here to fix some obvious misses
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% assuming 2 clusters (vte and nvte)
reclus = hclus;

nvtes = mode(hclus); % always find more nvtes than vtes in our data

% below won't work for more than 2 clusters!!
vtes = unique(hclus(hclus~=mode(hclus)));

tmins = zeros(1,size(traj_mat,2));
tmaxs = zeros(1,size(traj_mat,2));
for t = 1:size(traj_mat,2)
    txs = traj_mat(1:n,t);
    % creating logical array
    tmins(t) = min(txs(traj_mat(n+1:end,t)<0.75 & traj_mat(n+1:end,t)>0.4));
    tmaxs(t) = max(txs(traj_mat(n+1:end,t)<0.75 & traj_mat(n+1:end,t)>0.4));
end

% find trajectories that cross a position threshold
filtixs = (hclus==nvtes) & (tmins<0)' & (tmaxs-tmins>0.2)';
reclus(filtixs) = vtes;

% reassign VTEs that don't cross position threshold and have low zlnIdPhi
filtixs = (hclus==vtes) & (tmins<0 & tmins>-0.05)' & (tmaxs-tmins < 0.03)' &...
          (min(traj_mat(1:n,:))>-0.05)' & (zlnidphi<1);
reclus(filtixs) = nvtes;

% can also add thresholding based on low/high r^2 fit here if needed

% reassign initial non-VTEs that pass position threshold into VTE group
% these values require some fiddling and choices about what you do and
% don't consider a VTE (if you choose to use this step - can always skip)
filtixs = ((min(traj_mat(1:n,:)) < -0.04)' & reclus==nvtes) | ...
         ((tmaxs>0.2)' & (tmins < -0.03)' & reclus==nvtes);
reclus(filtixs) = vtes;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure
subplot(2,1,1)
scatter(proj_space(1,:),proj_space(2,:),20,...
             color_mat(hclus,:),'filled',...
             'MarkerFaceAlpha',0.75)
xlabel('PC1')
ylabel('PC2')
title('Original clustering')

subplot(2,1,2)
scatter(proj_space(1,:),proj_space(2,:),20,...
             color_mat(reclus,:),'filled',...
             'MarkerFaceAlpha',0.75)
xlabel('PC1')
ylabel('PC2')
title('Reassigned clustering')

%%
xgrid = repmat(repmat(0:2:2*9,n,1),1,10);
ygrid = [];
for ii = 0:9
    ygrid = [ygrid,ones(n,10)*ii*2];
end
randts = randsample(1:size(traj_mat,2),size(xgrid,2),false);
plot_mat = traj_mat(:,randts)+[xgrid;ygrid];
% clusrand = [reclus(randts);ones(size(xgrid,2)-size(plot_mat,2),1)];

f = figure;
hold on
for t = 1:size(plot_mat,2)
    plot(plot_mat(1:n,t),plot_mat(n+1:end,t),"Color", ...
        [color_mat(reclus(randts(t)),:) 0.5],'LineWidth',2)
end
hold off
xlim([-1 20])
ylim([-1 20])
ax = gca;
outerpos = ax.OuterPosition;
% ti = ax.TightInset;
ax.Position = outerpos;

xticks([])
yticks([])
f.Color = [0.981 1 0.911];
f.Units = 'inches';
f.Position = [1 2 10 8];



%% plot matrix
r = ceil(sqrt(height(bdata)));c = r-1;
m_ix = find(~ismember(1:size(traj_mat,2),dendp));
if ~isempty(m_ix)
    m_ix1 = find(Z(:,1)==m_ix);
    if isempty(m_ix1)
        m_ix1 = find(Z(:,2)==m_ix);
        near_t = Z(m_ix1,1);
    else
        near_t = Z(m_ix1,2);
    end
    if near_t>size(traj_mat,2)
        near_t = near_t-size(traj_mat,2);
    end
    near_ix = find(dendp==near_t);
    dendp = [dendp(1:near_ix),m_ix,dendp(near_ix+1:end)];
end
xgrid = repmat(repmat(0:2:2*c,n,1),1,r);
ygrid = [];
for ii = 0:c
    ygrid = [ygrid,ones(n,r)*ii*2];
end

plot_mat = [traj_mat(:,dendp),zeros(size(traj_mat,1),size(xgrid,2)-size(traj_mat,2))]+[xgrid;ygrid];
plotclus = [reclus(dendp);ones(size(xgrid,2)-size(traj_mat,2),1)];

f = figure;
% scatter(plot_mat(1:n,:),plot_mat(n+1:end,:), 7, color_mat(plotclus,:), ...
%         'filled','o','MarkerFaceAlpha',0.33)
hold on
for t = 1:size(plot_mat,2)
    plot(plot_mat(1:n,t),plot_mat(n+1:end,t),"Color", ...
        [color_mat(plotclus(t),:) 0.5],'LineWidth',1.5)
end
hold off

ylim([-1 2*r])
xlim([-1 2*c+2])

set(gca, 'Color', [0.981 1 0.941], 'XColor','none', 'YColor','none')
ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset;
ax.Position = outerpos;

xticks([])
yticks([])
f.Color = [0.981 1 0.911];
f.Units = 'inches';
f.Position = [1 2 10 8];

%% Click around PC space and plot trajectories
% this is useful, in my opinion! definitely click around here to see where
% you're getting false -/+ to inform your adjustments!

% d = 600; % reset d based on your data here if you want
hclus = cluster(Z,'cutoff',d,'criterion','distance');
fig = figure;
scatter(proj_space(1,:),proj_space(2,:),20,...
         color_mat(reclus,:),'filled',...
         'MarkerFaceAlpha',0.75)
xlabel('PC1')
ylabel('PC2')

datacursormode on
while isvalid(fig)
    disp('Click point to display a data tip')
    try
        waitforbuttonpress
        ax = gca();
        dtip = ax.Children.Children().DataIndex;
        display(dtip)
        figure
        plot(traj_mat(1:n,dtip),traj_mat(n+1:end,dtip), 'Color', ...
             [color_mat(reclus(dtip),:) 0.5],'LineWidth',2)
        xlim([-1 1])
        ylim([-0.1 1.5])
        axis off

    catch
        continue
    end
    if ~isvalid(fig)
        break
    end

end
close all

%% save VTE data
ID = string(string(bdata.rat_ID)+"_"+string(bdata.date));
bdata.ID = ID;
VTE = reclus;
VTE(VTE==mode(reclus)) = nan;
VTE(~isnan(VTE)) = 1;
VTE(isnan(VTE)) = 0;
%%
bdata.VTE = VTE;


for uID = unique(ID)'

    sesh = bdata(bdata.ID == uID,:);
    writetable(sesh,"VTE_table_test_"+uID+".csv")
end

%%
function cplot = coph_plot(tdata,max_d)
[~,S,V] = svd(tdata);
pspace = S*V';
cplot = nan(max_d,1);
for d = 1:max_d
    Z = linkage(pspace(1:d,:)','ward');
    cplot(d) = cophenet(Z,pdist(pspace(1:d,:)'));
end
end

function clickdata(ax)
dtip = ax.Children.Children().DataIndex;
display(dtip)
end

function lnidphi = calc_idphi(pos_mat)
    % pos_mat needs to by an m x 2 array with X and Y position data
    % expected to be combined with a splitapply to calculate zlnidphi
    
    % change in angular velocity of motion (Phi)
    dphi = diff(atan2(diff(pos_mat(:,2)), diff(pos_mat(:,1))));
    % integrated absolute dPhi
    lnidphi = log1p(sum(abs(dphi)));
end