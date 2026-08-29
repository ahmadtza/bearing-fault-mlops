clear
clc

load('MachineData.mat')

% ============================================================
% 1. Convert the 40 recordings to numerical matrices
%
% Result:
% 70000 rows = time samples
% 40 columns = recordings
% ============================================================

ch1 = cat(2, trainData.ch1{:});
ch2 = cat(2, trainData.ch2{:});
ch3 = cat(2, trainData.ch3{:});


% ============================================================
% 2. Create recording IDs
% ============================================================

run_id = (1:height(trainData))';


% ============================================================
% 3. Convert categorical labels to numerical labels
%
% 0 = After  = Normal
% 1 = Before = Anomalous
% ============================================================

label_code = double( trainData.label == 'Before');


% ============================================================
% 4. Display information
% ============================================================

disp('Size of ch1:')
disp(size(ch1))

disp('Size of ch2:')
disp(size(ch2))

disp('Size of ch3:')
disp(size(ch3))

disp('Run IDs:')
disp(run_id)

disp('Label codes:')
disp(label_code)


% ============================================================
% 5. Show class counts
% ============================================================

disp('Categories:')
disp(categories(trainData.label))

disp('Counts:')
disp(countcats(trainData.label))


% ============================================================
% 6. Save in Python-friendly MATLAB v7.3 format
% ============================================================

save('MachineData_export.mat', ...
    'ch1', ...
    'ch2', ...
    'ch3', ...
    'run_id', ...
    'label_code', ...
    '-v7.3')