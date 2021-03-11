% Megan Freer
% AMATH 482
% Assignment 2

%% Part 1: Sweet Child O' Mine by Guns N' Roses (guitar)
% in D♭ major
clear all; close all; clc;

% import and convert GNR (Sweet Child O' Mine)
[y, Fs] = audioread('GNR.m4a');

% Make computing faster (decreases sampling rate)
y = downsample(y,4);
Fs = Fs/4;

% Defining time vector, length of audio clip, and number of samples
t = (1:length(y))/Fs; % time points vector
clip_length = length(y)/Fs; % audio length in seconds
n = length(y); % number of samples

% Defining frequency vector and shifting zero-frequency component to center
% of spectrum
k = (1/clip_length)*[0:(n-1)/2 -(n/2):-1];
ks = fftshift(k);

% Gabor filtering (define window width and center of window tau)
a = 100; % used for Gabor filter
a_overtone = 0.01; % used for overtone filter (Gaussian filter)
num_notes = 57; % manually counted
windows_per_note = 5; % want multiple windows per note
tau = 0:(clip_length/num_notes)/windows_per_note:clip_length;

% Initializing matrices being stored
max_freq = zeros(length(tau), 1);
y_no_overtone_spec = zeros(length(y), length(tau));

for j = 1:length(tau)
    
    % Make Gabor Filter and apply to audio
    gabor_filter = exp(-a*(t - tau(j)).^2);
    y_gabor = gabor_filter.*y';
    
    % Transform audio to frequency domain
    y_gabor_freq = fft(y_gabor);
    
    % At each time point, find max amplitude and associated frequency
    [max_amp, index] = max(abs(y_gabor_freq));
    max_freq(j) = k(index);
    
    % Filter out overtone (in frequency domain)
    % Uses Gausian filter around center frequency
    overtone_filter = exp(-a_overtone*(k - k(index)).^2);
    no_overtone_freq = overtone_filter.*y_gabor_freq;
    
    % Getting ready to plot (shifting zero-frequency component to center of
    % spectrum and adding frequency data to spectrogram)
    y_no_overtone_spec(:,j) = fftshift(abs(no_overtone_freq));
end

% Plot spectrogram
figure()
pcolor(tau,ks,log(y_no_overtone_spec + 1))
shading interp; colormap(hot); colorbar;
ylim([250 800])
set(gca,'Fontsize',16)
xlabel('time (sec)'), ylabel('frequency (Hz)')
title("Sweet Child O' Mine by Guns N' Roses (guitar)");

% Plot "spectrogram" (music score) with max frequency at each timepoint
figure()
scatter(tau, max_freq)
ylim([250 800])
set(gca,'Fontsize',16)
xlabel('time (sec)'), ylabel('Notes')
title("Sweet Child O' Mine by Guns N' Roses (guitar)");
freq_plot = [277.18, 311.13, 349.23, 369.99, 415.3, 466.16, 523.25,...
             554.37, 622.25, 698.46, 739.99, 830.61];
yticks(freq_plot);
yticklabels({'D♭', 'E♭', 'F', 'G♭', 'A♭', 'B♭', 'C', 'D♭', 'E♭', 'F',...
             'G♭', 'A♭'});
for i = 1:length(freq_plot)
    yline(freq_plot(i));
end
%% Part 2: Comfortably Numb by Pink Floyd (bass)
% in B minor
clear all; close all; clc;

% import and convert Floyd (Comfortably Numb)
[y, Fs] = audioread('Floyd.m4a');

% Shorten to first 30 sec of audio clip for faster computing
y = y(1:length(y)/2);

% Make computing faster (decreases sampling rate)
y = downsample(y, 4);
Fs = Fs/4;

% Defining time vector, length of audio clip, and number of samples
t = (1:length(y))/Fs;  % time points vector
clip_length = length(y)/Fs; % audio length in seconds
n = length(y); % number of samples

% Defining frequency vector and shifting zero-frequency component to center
% of spectrum
k = (1/clip_length)*[0:(n-1)/2 -(n/2):-1];
ks = fftshift(k);

% Find the index of k for 60 Hz and 250 Hz (bass frequencies)
array_60_Hz = find(k > 60);
index_60_Hz = array_60_Hz(1);
array_250_Hz = find(k > 250);
index_250_Hz = array_250_Hz(1);

% Gabor filtering (define window width and center of window tau)
a = 100; % used for Gabor filter
a_overtone = 0.1; % used for overtone filter (Gaussian filter)
step_size = 0.1;
tau = 0:step_size:clip_length;

% Initializing matrices being stored
max_freq = zeros(length(tau), 1);
y_no_overtone_spec = zeros(length(y), length(tau));

for j = 1:length(tau)
    
    % Make Gabor Filter and apply to audio
    gabor_filter = exp(-a*(t - tau(j)).^2);
    y_gabor = gabor_filter.*y';
    
    % Transform audio to frequency domain
    y_gabor_freq = fft(y_gabor);
    
    % Filter to only include bass frequencies (60-250 Hz)
    y_gabor_freq(1:index_60_Hz - 1) = 0;
    y_gabor_freq(index_250_Hz + 1:length(y_gabor_freq)) = 0;
    
    % At each time point, find max amplitude and associated frequency
    [max_amp, index] = max(abs(y_gabor_freq));
    max_freq(j) = k(index);
    
    % Filter out overtone (in frequency domain)
    % Uses Gaussian filter around center frequency
    overtone_filter = exp(-a_overtone*(k - k(index)).^2);
    no_overtone_freq = overtone_filter.*y_gabor_freq;
    
    % Getting ready to plot (shifting zero-frequency component to center of
    % spectrum and adding frequency data to spectrogram)
    y_no_overtone_spec(:,j) = fftshift(abs(no_overtone_freq));
   
end

% Plot spectrogram
figure()
pcolor(tau,ks,log(y_no_overtone_spec + 1))
shading interp; colormap(hot); colorbar;
ylim([80 250])
set(gca,'Fontsize',16)
xlabel('time (sec)'), ylabel('frequency (Hz)')
title("Comfortably Numb by Pink Floyd (bass)");

% Plot "spectrogram" (music score) with max frequency at each timepoint
figure()
scatter(tau, max_freq)
ylim([80 250])
set(gca,'Fontsize',16)
xlabel('time (sec)'), ylabel('Notes')
title("Comfortably Numb by Pink Floyd (bass)");
freq_plot = [82.41, 92.5, 98, 110, 123.47, 138.59, 146.83, 164.81, 185,...
             196, 220, 246.94];
yticks(freq_plot);
yticklabels({'E', 'F#', 'G', 'A', 'B', 'C#', 'D', 'E', 'F#', 'G', 'A',...
             'B'});
for i = 1:length(freq_plot)
    yline(freq_plot(i));
end
%% Part 3: Comfortably Numb by Pink Floyd (guitar)
% in B minor
clear all; close all; clc;

% import and convert Floyd (Comfortably Numb)
[y, Fs] = audioread('Floyd.m4a');

% Shorten to first 10 sec of audio clip for faster computing
y = y(1:length(y)/2);

% Make computing faster (decreases sampling rate)
y = downsample(y, 8);
Fs = Fs/8;

% Defining time vector, length of audio clip, and number of samples
t = (1:length(y))/Fs; % time points vector
clip_length = length(y)/Fs; % audio total length in seconds
n = length(y); % number of samples

% Defining frequency vector and shifting zero-frequency component to center
% of spectrum
k = (1/clip_length)*[0:(n-1)/2 -(n/2):-1];
ks = fftshift(k);

% Find the index of k for 250 Hz (maximum bass frequency)
array_250_Hz = find(k > 250);
index_250_Hz = array_250_Hz(1);

% Gabor filtering (define window width and center of window tau)
a = 100; % used for Gabor filter
a_overtone_guit = 0.1; % used for guitar overtone filter (Gaussian filter)
step_size = clip_length/200;
tau = 0:step_size:clip_length;

% Initializing matrices being stored
max_freq_guitar = zeros(length(tau), 1);
y_no_overtone_spec = zeros(length(y), length(tau));

for j = 1:length(tau)
    
    % Make Gabor Filter and apply to audio
    gabor_filter = exp(-a*(t - tau(j)).^2);
    y_gabor = gabor_filter.*y';
    
    % Transform audio to frequency domain
    y_gabor_freq = fft(y_gabor);

    % Apply band pass filter to get rid of bass (get rid of below 250 Hz)
    no_bass_overtone_freq = y_gabor_freq;
    no_bass_overtone_freq(1:index_250_Hz) = 0;
    
    % Remove amplitudes of all negative frequencies
    no_bass_overtone_freq(length(y_gabor_freq)/2:length(y_gabor_freq)) = 0;  
    
    % At each time point, find max amplitude and associated frequency of
    % guitar (center frequency)
    [max_amp_guitar, index_guitar] = max(abs(no_bass_overtone_freq));
    max_freq_guitar(j) = k(index_guitar);
    
    % Filter out overtones of guitar (in frequency domain)
    overtone_filter_guitar = exp(-a_overtone_guit*(k-k(index_guitar)).^2);
    y_guitar = overtone_filter_guitar.*no_bass_overtone_freq;
    
    % Getting ready to plot (shifting zero-frequency component to center of
    % spectrum and adding frequency data to spectrogram)
    y_no_overtone_spec(:,j) = fftshift(abs(y_guitar));
   
end

% Plot spectrogram
figure()
pcolor(tau,ks,log(y_no_overtone_spec + 1))
shading interp; colormap(hot); colorbar;
ylim([0 1000])
set(gca,'Fontsize',16)
xlabel('time (sec)'), ylabel('frequency (Hz)')
title("Comfortably Numb by Pink Floyd (guitar)");

% Plot "spectrogram" (music score) with max frequency at each timepoint
figure()
scatter(tau, max_freq_guitar)
ylim([250 1000])
set(gca,'Fontsize',16)
xlabel('time (sec)'), ylabel('Notes')
title("Comfortably Numb by Pink Floyd (guitar)");
freq_plot = [329.63, 369.99, 392, 440, 493.88, 554.37, 587.33, 659.25,...
             739.99, 783.99, 880, 987.77];
yticks(freq_plot);
yticklabels({'E', 'F#', 'G', 'A', 'B', 'C#', 'D', 'E', 'F#', 'G', 'A',...
             'B'});
for i = 1:length(freq_plot)
    yline(freq_plot(i));
end