% Megan Freer
% AMATH 482
% Assignment 4

clear all; close all; clc;

%% Load in and reshape data
% load in the MNIST training data
[training_images, training_labels] = mnist_parse('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');

% load in the MNIST test data
[test_images, test_labels] = mnist_parse('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte');

% reshape each image into a column vector and each column of data matrix is
% a different image
training_images_col = reshape(training_images, [size(training_images, 1)*...
                              size(training_images, 2), size(training_images, 3)]);
training_images_col = im2double(training_images_col);

test_images_col = reshape(test_images, [size(test_images, 1)*size(test_images, 2),...
                          size(test_images, 3)]);
test_images_col = im2double(test_images_col);

%% 1) De-mean
training_mean = mean(training_images_col, 2); % column vector w/ mean of each row (784x1)
training_images_col = training_images_col - training_mean;
test_images_col = test_images_col - training_mean;

%% 2) Divide by largest singular value
[U_train,S_train,V_train] = svd(training_images_col, 'econ');
[U_test,S_test,V_test] = svd(test_images_col, 'econ');
largest_singular_value = S_train(1,1);
training_images_col = training_images_col / largest_singular_value;
test_images_col = test_images_col / largest_singular_value;

%% Figure 1: Sigmas

% Finding sigma values from S matrix (with training data)
sigmas_train = diag(S_train);

% Plot sigmas (with training data)
figure()
scatter(1:length(S_train), sigmas_train, 's')
title('Training Sigmas', 'FontSize', 18)
xlabel('Sigma Number', 'FontSize', 18)
ylabel('Sigma Value', 'FontSize', 18)
set(gca, 'FontSize', 14)

%% Figure 2: Sigma Energies

% Calculate energy for each rank/sigma (with training data)
energies_train = zeros(1, length(sigmas_train));
for i=1:length(sigmas_train)
    energies_train(i) = sum(sigmas_train(1:i).^2)/sum(sigmas_train.^2);
end

% Plot energies (with training data)
figure()
bar(1:length(energies_train), energies_train)
ylim([0 1])
title('Training Sigma Energies', 'FontSize', 18)
xlabel('Rank', 'FontSize', 18)
ylabel('Energy', 'FontSize', 18)
set(gca, 'FontSize', 14)

%% Figure 3: First 9 Principal Components

figure()
for i = 1:9
    subplot(3,3,i)
    ut1 = reshape(U_train(:,i), size(training_images, 1), size(training_images, 2));
    ut2 = rescale(ut1);
    imshow(ut2)
    title('Principal Component ' + string(i))
end

%% Figure 4: Projections Onto Principal Components 2, 3, and 5

projection = S_train*V_train'; % projection onto principal components: X = USV' --> U'X = SV'
projection = projection';
projection = projection(:, [2 3 5]); % 60,000 images x 3 modes
% x = column 2, y = column 3, z = column 5

figure()
colormap = brewermap(max(training_labels+1),'Set1'); % Creating the colormap
markersize = 20;
colors = colormap(training_labels+1); % maps each label to the corresponding color
scatter3(projection(:,1), projection(:,2), projection(:,3), markersize, colors)
title('Training Data Projections Onto V-modes Colored By Digit Label', 'FontSize', 16)
xlabel('Principal Component 2', 'FontSize', 16)
ylabel('Principal Component 3', 'FontSize', 16)
zlabel('Principal Component 5', 'FontSize', 16)
set(gca, 'FontSize', 14)

%% Deciding if separating 2 or 3 digits
num_digits_separating = 2;

if num_digits_separating == 2
    %% Two Digit Recognition: narrowing down data (training data)
    
    % Choosing numbers using
    num_modes = 50;
    num_a = 6;
    num_b = 7;
    
    % narrow down to only values testing (training data)
    indexes_a = find(training_labels == num_a);
    indexes_b = find(training_labels == num_b);
    combined = [indexes_a; indexes_b];
    indexes_train = sort(combined);
    
    % narrow labels to only include two numbers looking at (training data)
    training_labels = training_labels(indexes_train);
    
    % narrow pictures to only include two numbers looking at (training data)
    training_images_col = training_images_col(:, indexes_train);
    
    % narrow down to only values testing (test data)
    indexes_a = find(test_labels == num_a);
    indexes_b = find(test_labels == num_b);
    combined = [indexes_a; indexes_b];
    indexes_test = sort(combined);
    
    % narrow labels to only include two numbers looking at (test data)
    test_labels = test_labels(indexes_test);
    
    % narrow labels to only include two numbers looking at (test data)
    test_images_col = test_images_col(:, indexes_test);
    
    %% 3) Project onto PCA modes (V-modes) (training data)
    % do projection with training data
    projection_train = S_train*V_train'; % projection onto principal components: X = USV' --> U'X = SV'
    
    % narrow training data to first 50 modes and images that are 0 or 1
    projection_train = projection_train([1:num_modes], indexes_train);
    projection_train = projection_train';
    
    % do projection with test data
    projection_test = S_test*V_test';
    
    % narrow test data to first 50 modes and images that are 0 or 1
    projection_test = projection_test([1:num_modes], indexes_test);
    projection_test = projection_test';
    
    %% 4) Use classify (for LDA) (on test data)
    class = classify(projection_test, projection_train, training_labels);
    
    wrong_count = 0;
    for i = 1:length(class)
        if class(i) ~= test_labels(i)
            wrong_count = wrong_count + 1;
        end
    end
    
    accuracy_lda_2_digits = (1-(wrong_count/length(class)))*100 % in percents
    
    %% Use classify (for LDA) (on training data)
    class = classify(projection_train, projection_train, training_labels);
    
    wrong_count = 0;
    for i = 1:length(class)
        if class(i) ~= training_labels(i)
            wrong_count = wrong_count + 1;
        end
    end
    
    accuracy_lda_2_digits_train = (1-(wrong_count/length(class)))*100 % in percents
    
    %% Classification tree on fisheriris data (on test data)
    load fisheriris;
    
    tree=fitctree(projection_train,training_labels);
    predicted_labels = predict(tree,projection_test);
    
    wrong_count = 0;
    for i = 1:length(predicted_labels)
        if predicted_labels(i) ~= test_labels(i)
            wrong_count = wrong_count + 1;
        end
    end
    
    accuracy_tree = (1-(wrong_count/length(predicted_labels)))*100 % in percents
    
    %% Classification tree on fisheriris data (on training data)
    load fisheriris;
    
    tree=fitctree(projection_train,training_labels);
    predicted_labels = predict(tree,projection_train);
    
    wrong_count = 0;
    for i = 1:length(predicted_labels)
        if predicted_labels(i) ~= training_labels(i)
            wrong_count = wrong_count + 1;
        end
    end
    
    accuracy_tree_train = (1-(wrong_count/length(predicted_labels)))*100 % in percents
    
    %% SVM classifier (for test data)
    svm_model = fitcsvm(projection_train,training_labels);
    testlabels_svm = predict(svm_model,projection_test);
    
    wrong_count = 0;
    for i = 1:length(testlabels_svm)
        if testlabels_svm(i) ~= test_labels(i)
            wrong_count = wrong_count + 1;
        end
    end
    
    accuracy_svm = (1-(wrong_count/length(testlabels_svm)))*100 % in percents
    
    %% SVM classifier (for training data)
    svm_model = fitcsvm(projection_train,training_labels);
    testlabels_svm = predict(svm_model,projection_train);
    
    wrong_count = 0;
    for i = 1:length(testlabels_svm)
        if testlabels_svm(i) ~= training_labels(i)
            wrong_count = wrong_count + 1;
        end
    end
    
    accuracy_svm_train = (1-(wrong_count/length(testlabels_svm)))*100 % in percents
elseif num_digits_separating == 3
    %% Three Digit Recognition: narrowing down data (training data)

    % Choosing numbers using
    num_modes = 50;
    num_a = 3;
    num_b = 4;
    num_c = 9;
    
    % narrow down to only values testing (training data)
    indexes_a = find(training_labels == num_a);
    indexes_b = find(training_labels == num_b);
    indexes_c = find(training_labels == num_c);
    combined = [indexes_a; indexes_b; indexes_c];
    indexes_train = sort(combined);
    
    % narrow labels to only include 3 numbers looking at (training data)
    training_labels = training_labels(indexes_train);
    
    % narrow pictures to only include 3 numbers looking at (training data)
    training_images_col = training_images_col(:, indexes_train);
    
    % narrow down to only include 3 numbers looking at (test data)
    indexes_a = find(test_labels == num_a);
    indexes_b = find(test_labels == num_b);
    indexes_c = find(test_labels == num_c);
    combined = [indexes_a; indexes_b; indexes_c];
    indexes_test = sort(combined);
    
    % narrow labels to only include 3 numbers looking at (test data)
    test_labels = test_labels(indexes_test);
    
    % narrow pictures to only include 3 numbers looking at (test data)
    test_images_col = test_images_col(:, indexes_test);
    
    %% 3) Project onto PCA modes (V-modes) (training data)
    % do projection with training data
    projection_train = S_train*V_train'; % projection onto principal components: X = USV' --> U'X = SV'
    
    % narrow training data to correct number modes and images that are 3 numbers
    projection_train = projection_train([1:num_modes], indexes_train);
    projection_train = projection_train'; % now 12,665x50
    
    % do projection with test data
    projection_test = S_test*V_test';
    
    % narrow test data to correct number modes and images that are 3 numbers
    projection_test = projection_test([1:num_modes], indexes_test);
    projection_test = projection_test';
    
    %% 4) Use classify (for LDA) (for test data)
    class = classify(projection_test, projection_train, training_labels);
    
    wrong_count = 0;
    for i = 1:length(class)
        if class(i) ~= test_labels(i)
            wrong_count = wrong_count + 1;
        end
    end
    
    accuracy_lda_3_digits = (1-(wrong_count/length(class)))*100 % in percents
    
    %% Use classify (for LDA) (for training data)
    class = classify(projection_train, projection_train, training_labels);
    
    wrong_count = 0;
    for i = 1:length(class)
        if class(i) ~= training_labels(i)
            wrong_count = wrong_count + 1;
        end
    end
    
    accuracy_lda_3_digits_train = (1-(wrong_count/length(class)))*100 % in percents
end