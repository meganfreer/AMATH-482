% Megan Freer
% AMATH 482
% Assignment 3

clear all; close all; clc;

% Note: Uncomment a section to create plots for that test set

% Test 1: Ideal Case
camera_names = ["cam1_1.mat", "cam2_1.mat", "cam3_1.mat"];
video_names = ["vidFrames1_1", "vidFrames2_1", "vidFrames3_1"];
test_title = 'Test 1: Ideal Case';

% % Test 2: Noisy Case
% camera_names = ["cam1_2.mat", "cam2_2.mat", "cam3_2.mat"];
% video_names = ["vidFrames1_2", "vidFrames2_2", "vidFrames3_2"];
% test_title = 'Test 2: Noisy Case';

% % Test 3: Horizontal Displacement (also swings side to side)
% camera_names = ["cam1_3.mat", "cam2_3.mat", "cam3_3.mat"];
% video_names = ["vidFrames1_3", "vidFrames2_3", "vidFrames3_3"];
% test_title = 'Test 3: Horizontal Displacement';

% % Test 4: Horizontal Displacement and Rotation
% camera_names = ["cam1_4.mat", "cam2_4.mat", "cam3_4.mat"];
% video_names = ["vidFrames1_4", "vidFrames2_4", "vidFrames3_4"];
% test_title = 'Test 4: Horizontal Displacement and Rotation';

number_cameras = 3;
positions = cell(1, 3);

for i = 1:number_cameras
    % Load in video
    load(camera_names(i));
    video = eval(video_names(i));
    
    % Finding number of frames (4th dimension of video)
    number_frames = size(video,4);
    
    % Find size in x and y direction (# pixels)
    pixels_x = size(video,1);
    pixels_y = size(video,2);
    
    % Initialize matrix to store grey scale frames
    grey_scale = zeros(pixels_x, pixels_y, number_frames, 'uint8');
    
    % Loop through frames and convert to grey scale
    for j = 1:number_frames
        grey_scale(:,:,j) = rgb2gray(video(:,:,:,j));
    end 
    
    % Have user select part of image to be studied
    figure()
    imshow(grey_scale(:,:,1))
    rect = getrect;
    
    % Initialize difference image and position data
    difference = zeros(pixels_x, pixels_y, 'uint8');
    position = zeros(2, number_frames-1);
    
    % Loop through each frame
    for j = 1:number_frames-1
        % Calculate difference between this frame and next frame to find
        % moving parts
        difference(:,:) = imabsdiff(grey_scale(:,:,j), grey_scale(:,:,j+1));
        
        % Extract information from rect
        x_min = round(rect(1));
        y_min = round(rect(2));
        x_max = round(rect(1)+rect(3));
        y_max = round(rect(2)+rect(4));
        
        % Binarize difference image
        binarized = imbinarize(difference(y_min:y_max, x_min:x_max), 0.3);
        
        % Returns all the (x, y) coordinates on the image that are white/1
        [x_coordinates, y_coordinates] = find(binarized);
        if length(x_coordinates) >= 1
            % Save the average location of the white space
            position(:, j) = [mean(x_coordinates), mean(y_coordinates)];
        end
        
        % Plot calculated position on top of image
        imshow(binarized)
        hold on
        plot(position(2,j), position(1,j), 'rp', 'MarkerFaceColor', 'red',...
             'MarkerSize', 24)
        hold off
        drawnow
    end
    
    % At this point, there are many zeros still in the position data
    % (black frames) -> need to replace these zeros
    frames_array = 1:length(position);
    zero_positions = find(position(1,:) == 0); % where to find estimates
    nonzero_positions = find(position(1,:) ~= 0);
    position_no0 = zeros(2, length(nonzeros(position(1,:)))); % resets size for each loop
    position_no0(1,:) = nonzeros(position(1,:));
    position_no0(2,:) = nonzeros(position(2,:));
    
    % need to remove 0s before entering into interp1
    position(1,zero_positions) = interp1(nonzero_positions, position_no0(1,:),...
             zero_positions);
    position(2,zero_positions) = interp1(nonzero_positions, position_no0(2,:),...
             zero_positions);
    
    % Make all position arrays start at the same point in time
    if i == 1 || i == 2
        % Find index of minimum in first 25 frames
        [starting_point, starting_index] = min(position(1,1:25));
    end
    
    % Third video is rotated -> find the minimum vertical position from
    % other dimension
    if i == 3
        % Find index of minimum in first 25 frames
        [starting_point, starting_index] = min(position(2,1:25));
    end
    
    % Crop the position data to the correct starting point
    position = position(:, starting_index:length(position));
    
    % Remove any NaN values
    position = rmmissing(position, 2); % means removing column
    
    % Save data adjusted to correct start point to a cell
    positions{i} = position;
end

% Find minimum length of the previous position arrays
minimum_length = min(cellfun('size', positions, 2));

% Loop through each camera/position array and make the same length
for i = 1:number_cameras
    positions{i} = positions{i}(:, 1:minimum_length);
end

% Make X (matrix with all (x,y) data for 3 cameras)
% For PCA to work properly, subtract mean from each of the data dimensions
X = zeros(6, minimum_length);
X(1,:) = positions{1}(2,:) - mean(positions{1}(2,:)); % x1
X(2,:) = positions{1}(1,:) - mean(positions{1}(1,:)); % y1
X(3,:) = positions{2}(2,:) - mean(positions{2}(2,:)); % x2
X(4,:) = positions{2}(1,:) - mean(positions{2}(1,:)); % y2
X(5,:) = positions{3}(2,:) - mean(positions{3}(2,:)); % x3
X(6,:) = positions{3}(1,:) - mean(positions{3}(1,:)); % y3

% Plot horizontal positions
figure()
scatter(1:length(X), X(1,:), 100, '.'); hold on
scatter(1:length(X), X(3,:), 100, '.'); hold on
scatter(1:length(X), X(6,:), 100, '.'); hold on
legend('Camera 1', 'Camera 2', 'Camera 3')
xlabel('Video Frame', 'FontSize', 18)
ylabel('Horizontal Position', 'FontSize', 18)
title(test_title, 'FontSize', 18)
set(gca, 'FontSize', 14)

% Plot vertical positions
figure()
scatter(1:length(X), X(2,:), 100, '.'); hold on
scatter(1:length(X), X(4,:), 100, '.'); hold on
scatter(1:length(X), X(5,:), 100, '.'); hold on
legend('Camera 1', 'Camera 2', 'Camera 3')
xlabel('Video Frame', 'FontSize', 18)
ylabel('Vertical Position', 'FontSize', 18)
title(test_title, 'FontSize', 18)
set(gca, 'FontSize', 14)

% Actual PCA part
[U,S,V] = svd(X, 'econ');

% Analyzing sigmas
sig = diag(S);

% Calculate energy for each rank/sigma
energy1 = sig(1)^2/sum(sig.^2);
energy2 = sum(sig(1:2).^2)/sum(sig.^2);
energy3 = sum(sig(1:3).^2)/sum(sig.^2);
energy4 = sum(sig(1:4).^2)/sum(sig.^2);
energy5 = sum(sig(1:5).^2)/sum(sig.^2);
energy6 = sum(sig(1:6).^2)/sum(sig.^2);
energies = [energy1, energy2, energy3, energy4, energy5, energy6];

% Plot sigmas
figure()
bar(1:6, sig, 'c')
title(test_title, 'FontSize', 18)
xlabel('Sigma Number', 'FontSize', 18)
ylabel('Sigma Value', 'FontSize', 18)
set(gca, 'FontSize', 14)

% Plot energies
figure()
bar(1:6, energies, 'c')
ylim([0 1])
text(1:length(energies),energies,num2str(energies'),'vert','top','horiz',...
    'center', 'FontSize', 12); 
title(test_title, 'FontSize', 18)
xlabel('Rank', 'FontSize', 18)
ylabel('Energy', 'FontSize', 18)
set(gca, 'FontSize', 14)
