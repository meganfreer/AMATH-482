% Megan Freer
% AMATH 482
% Assignment 1

%% Code from Assignment Introduction
% Clean workspace
clear all; close all; clc

% Imports the data as the 262144x49 (space by time) matrix called subdata
load subdata.mat 

L = 10; % spatial domain
n = 64; % Fourier modes
x2 = linspace(-L,L,n+1); 
x = x2(1:n); y=x; z=x;
k = (2*pi/(2*L))*[0:(n/2 - 1) -n/2:-1]; 
ks = fftshift(k);

[X,Y,Z]=meshgrid(x,y,z);
[Kx,Ky,Kz]=meshgrid(ks,ks,ks);

for t=1:49
    Un(:,:,:) = reshape(subdata(:,t),n,n,n);
    M = max(abs(Un),[],'all');
    % isosurface(X,Y,Z,abs(Un)/M,0.7)
    % axis([-20 20 -20 20 -20 20]), grid on
    % drawnow
    % pause(1)
end

%% Part 1: averaging spectrum -> frequency signature (center frequency)

% Intializing sum of frequencies over all time points
sum_freq = zeros(n, n, n);

% Loop over all time points and sum data in frequency domain
for t=1:49
    Un(:, :, :) = reshape(subdata(:,t), n, n, n);
    sum_freq = sum_freq +  fftn(Un); % adds the Fourier transform
end

% Find average from sum of frequncies
average_freq = abs(sum_freq) / 49;

% Find max frequency component (center frequency) and index of max freq
[max_freq, max_index] = max(average_freq, [], 'all', 'linear');

% Use linear index from above to find 64x64x64 indices
[I1, I2, I3] = ind2sub([n n n], max_index);

% Finding frequency center in 3D frequency domain
Kx_center = Kx(I1, I2, I3);
Ky_center = Ky(I1, I2, I3);
Kz_center = Kz(I1, I2, I3);

% Combining x, y, and z components of center frequency
center_freq = abs(([Kx_center, Ky_center, Kz_center]));

%% Part 2: filter data around center freq -> denoise data -> submarine path

% Create Gaussian spectral filter
tau = 0.5;
filter_x = exp(-tau .* (Kx - Kx_center) .^ 2);
filter_y = exp(-tau .* (Ky - Ky_center) .^ 2);
filter_z = exp(-tau .* (Kz - Kz_center) .^ 2);
filter = filter_x .* filter_y .* filter_z;

locations = zeros(3, 49);
figure()

% Looping through all time points
for t = 1:49
    Un(:, :, :) = reshape(subdata(:,t), n, n, n);
    
    % Change to frequency domain
    unfiltered_Un_freq = fftn(Un);
    
    % Apply filter to signal in frequency space
    filtered_Un_freq = filter .* unfiltered_Un_freq;
    
    % Go back to spacial domain
    % (now ifftn, not ifft like in lecture)
    filtered_Un_space = ifftn(filtered_Un_freq);
    
    % Find the index of the max frequency at each timepoint
    [max_freq, max_index] = max(abs(filtered_Un_space), [], 'all', 'linear');
    
    % Use linear index from above to find 64x64x64 indices
    [I1, I2, I3] = ind2sub([n n n], max_index);
    
    % Save in locations
    locations(1,t) = X(I1, I2, I3);
    locations(2,t) = Y(I1, I2, I3);
    locations(3,t) = Z(I1, I2, I3);
    
    % Make a plot of the saved locations
    plot3(locations(1,t), locations(2,t), locations(3,t), '*r')
    hold on
end

% Label the figure
title('Submarine Path over 24-hour Period', 'FontSize', 24)
xlabel('X coordinate', 'FontSize', 20)
ylabel('Y coordinate', 'FontSize', 20)
zlabel('Z coordinate', 'FontSize', 20)
set(gca, 'FontSize', 16)

%% Part 3: give x and y coordinates in a table to follow the submarine

x_coordinate = locations(1,:)';
y_coordinate = locations(2,:)';
x_y_coordinates = table(x_coordinate, y_coordinate);