clear;
close all;

% load dataset to data variable
datapath = 'data3.mat';
load(datapath); 

figure;
h1 = gscatter(data(:,1), data(:,2));
hold on;
x1 = min(data(:,1))-2:0.1:max(data(:,1))+2;
x2 = min(data(:,2)):0.1:max(data(:,2));
[X1,X2] = meshgrid(x1,x2);

set(gca, 'color', [0.3 0.5 0.6])
set(gcf, 'color', [0.3 0.5 0.6])

% create a gif from the plotting option
gif = true;
if gif
    filename = 'gmm.gif';
    delete(filename);
end
%% initialization step
% declare parameters: all mu(s) and sigma(s), and phi(s)
K = 3; % choose number of clusters
gammas = zeros(size(data, 1), K);
mus = zeros(K, size(data, 2)); % matrix of size K x D
phi = ones(K, 1) / K;
sigmas = struct(); % structure that store covariance matrices of size D x D

% search for a good initialization values using
% some random sampled example
for ii = 1:K
    samples = data(randsample(1:size(data,1), 1), :);
    mus(ii, :) = samples;
    sigmas(ii).covmat = 0.1 * rand(1,1) * cov(data);
    
    % plot the initial mixture models
    F = mvnpdf([X1(:) X2(:)],mus(ii, :),sigmas(ii).covmat);
    F = reshape(F,length(x2),length(x1));
    [ct, hn(ii)] = contour(x1,x2,F,[.0001 .001 .01 .05:.1:.95 .99 .999 .9999]); 
    hmus(ii) = plot(mus(ii,1),mus(ii,2),'kx','LineWidth',2,'MarkerSize',10);

    frame = getframe(1);
    im = frame2im(frame);
    [imind,cm] = rgb2ind(im,256);
    imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
end

%% EM iteration steps
epsilon = 0.001;
iteration = 1;
loglikelihood = [];
while(true)
    % Expectitation steps
    % Update gammas
    for i = 1:size(data,1)
        % compute the new values of gammas
        newgammas = arrayfun(@(k) phi(k) * mvnpdf(data(i,:), ...
                    mus(k, :), sigmas(k).covmat), 1:K);
        % normalize them
        newgammas = newgammas / sum(newgammas);
        gammas(i, :) = newgammas;
    end
    
    % Maximization steps
    % Estimate mu(s), sigma(s), and covariance matrix
    Ns = sum(gammas);
    for j = 1:K
        % update j-th mu
        for d = 1:size(data,2) % iterate through data dimension
            mus(j, d) = sum(arrayfun(@(i) gammas(i, j) * ...
                            data(i, d), 1:size(data, 1)))/Ns(j);
        end
        
        % update j-th covariance matrix
        sigmas(j).covmat = ((data(:,:)-mus(j,:))' * ...
                            (gammas(:, j) .* (data(:,:)-mus(j,:))))/Ns(j);
        
        % update j-th phi
        phi = (Ns/sum(Ns))';
        
        delete(hn(j));
        delete(h1);
        delete(hmus(j));
        
        [M,I] = max(gammas, [], 2);
        h1 = gscatter(data(:,1), data(:,2), I);
        
        F = mvnpdf([X1(:) X2(:)],mus(j, :),sigmas(j).covmat);
        F = reshape(F,length(x2),length(x1));
        [ct, hn(j)] = contour(x1,x2,F,[.0001 .001 .01 .05:.1:.95 .99 .999 .9999]); 
        
        hmus(j) = plot(mus(j,1),mus(j,2),'kx','LineWidth',2,'MarkerSize',10);
        uistack(h1, 'bottom');
        drawnow;
        
        frame = getframe(1);
        im = frame2im(frame);
        [imind,cm] = rgb2ind(im,256);
        imwrite(imind,cm,filename,'gif','WriteMode','append');
    end
    
    % compute log likelihood
    % plot the progress
    laccum = 0;
    for i = 1:size(data,1)
        i_laccum = 0;
        for j = 1:K
            i_laccum = i_laccum + phi(j)* mvnpdf(data(i,:), mus(j, :), sigmas(j).covmat);
        end
        laccum = laccum + log(i_laccum);
    end
    fprintf('[%d-th iteration] log-likelihood: %.3f\n', iteration, laccum);
    loglikelihood = [loglikelihood; laccum];
    iteration = iteration + 1;
    
    % stopping condition
    % case when log-likelihood stop improving by more than eps
    if numel(loglikelihood) > 1
        if abs(loglikelihood(end)-loglikelihood(end-1)) <= epsilon
            fprintf('[Optimization completed]\n');
            break;
        end
    end
end