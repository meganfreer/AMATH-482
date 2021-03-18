% Megan Freer
% AMATH 482
% Assignment 5

clear all; close all; clc;

% Uncomment below to analyze the other video
v = VideoReader('ski_drop_low.mp4');
% v = VideoReader('monte_carlo_low.mp4');

% Read all video frames from file associated with v
frames = read(v);
height = size(frames, 1);
width = size(frames, 2);
timepoints = round(size(frames, 4) / 4); % downsample bc computer issues

% Make matrix to save black and white frames in
bw_frames = zeros(height, width, timepoints);

% Loop through and make black and white
for i = 1:timepoints
    current_frame = frames(:,:,:,i);
    bw_frame = rgb2gray(current_frame);
    bw_frames(:, :, i) = bw_frame;
end

% Reshape each frame into a column
bw_frames_2d = reshape(bw_frames, [height*width, timepoints]);

% Converts the image to double precision
bw_frames_2d = im2double(bw_frames_2d);

% Find dt and time vector
dt = 1/v.FrameRate;
t = 0:dt:timepoints-1;

% Create DMD matrices
X1 = bw_frames_2d(:,1:end-1);
X2 = bw_frames_2d(:,2:end);

% Perform SVD of X1
[U, Sigma, V] = svd(X1,'econ');

% Finding sigma values from S matrix
sigmas = diag(Sigma);
num_modes_plotting = 20;

% Plot sigmas
figure()
scatter(1:num_modes_plotting, sigmas(1:num_modes_plotting), 's')
title('Singular Values', 'FontSize', 18)
xlabel('Sigma Number', 'FontSize', 18)
ylabel('Sigma Value', 'FontSize', 18)
set(gca, 'FontSize', 14)

% Calculate energy for first 20 rank/sigma
energies = zeros(1, num_modes_plotting);
for i=1:num_modes_plotting
    energies(i) = sum(sigmas(1:i).^2)/sum(sigmas.^2);
end

% Plot energies
figure()
bar(1:num_modes_plotting, energies)
ylim([0 1])
title('Sigma Energies', 'FontSize', 18)
xlabel('Rank', 'FontSize', 18)
ylabel('Energy', 'FontSize', 18)
set(gca, 'FontSize', 14)

% Choose smaller number of modes
modes = 2;
U_small = U(:, 1:modes);
Sigma_small = Sigma(1:modes, 1:modes);
V_small = V(:, 1:modes);

% Perform computation of S~
S = U_small'*X2*V_small*diag(1./diag(Sigma_small));
[eV, D] = eig(S); % compute eigenvalues + eigenvectors
mu = diag(D); % extract eigenvalues

omega = log(mu)/dt;
Phi = U_small*eV;

% Create DMD Solution
y0 = Phi\X1(:,1); % pseudoinverse to get initial conditions
u_modes = zeros(length(y0),timepoints - 1);
for iter = 1:timepoints - 1
   u_modes(:,iter) = y0.*exp(omega*t(iter)); 
end
u_dmd = Phi*u_modes;

% Create sparse and low-rank matrixes
X_DMD_sparse = X1 - abs(u_dmd);
R = X_DMD_sparse .* (X_DMD_sparse < 0); % residual negative values matrix

% Find the minimum value in order to add into the sparse data later
min_value = abs(min(min(X_DMD_sparse)));

% Use residual negative values matrix
X_DMD_low_rank = R + abs(u_dmd); % background
X_DMD_sparse = X_DMD_sparse + min_value; % suggestion instead of subtracting R

% Create background and foreground videos
background = reshape(u_dmd, [height, width, timepoints-1]);
foreground = reshape(X_DMD_sparse, [height, width, timepoints-1]);

% Plot a resulting video frame
figure()
imshow(uint8(background(:,:,35)))
title('Background (Frame 35)', 'FontSize', 20)

figure()
imshow(uint8(foreground(:,:,35)))
title('Foreground (Frame 35)', 'FontSize', 20)

% Play videos
figure()
for i = 1:timepoints - 1
    imshow(uint8(foreground(:,:,i)))
    title('Foreground Video', 'FontSize', 20)
    hold off
    drawnow   
end

figure()
for i = 1:timepoints - 1
    imshow(uint8(background(:,:,i)))
    title('Background Video', 'FontSize', 20)
    hold off
    drawnow   
end