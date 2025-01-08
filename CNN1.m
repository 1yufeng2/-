%% 清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%% 导入数据
res = xlsread('8 phosphor data sets.xlsx'); % 假设扩展后的数据包含960个样本

%% 划分训练集、验证集和测试集
temp = randperm(960);  % 生成乱序数组，样本数为960

% 按比例 3:1:1 划分数据集
P_train = res(temp(1:576), 1:20)';         % 训练集 60% (576个样本)
T_train = res(temp(1:576), 21)';
P_val   = res(temp(577:768), 1:20)';       % 验证集 20% (192个样本)
T_val   = res(temp(577:768), 21)';
P_test  = res(temp(769:end), 1:20)';       % 测试集 20% (192个样本)
T_test  = res(temp(769:end), 21)';

M = size(P_train, 2); 
V = size(P_val, 2);
N = size(P_test, 2); 

%% 数据归一化
[P_train, ps_input] = mapminmax(P_train, 0, 1);  % 训练集归一化
P_val = mapminmax('apply', P_val, ps_input);      % 验证集归一化
P_test = mapminmax('apply', P_test, ps_input);    % 测试集归一化

t_train = categorical(T_train)';
t_test = categorical(T_test)';
t_val = categorical(T_val)';

%% 数据平铺
p_train = double(reshape(P_train, 20, 1, 1, M));
p_val   = double(reshape(P_val  , 20, 1, 1, V));
p_test  = double(reshape(P_test , 20, 1, 1, N));

%% 定义网络结构
layers = [
    imageInputLayer([20 1 1]) % 输入层，假设输入数据的尺寸是 20x1x1
    
    convolution2dLayer([2, 1], 32, 'Padding', 'same')  % 卷积层，大小为 2x1，32个滤波器
    batchNormalizationLayer() % 批量归一化层
    reluLayer() % 激活函数 ReLU
    dropoutLayer(0.2) % Dropout层，丢弃20%的神经元
    
    convolution2dLayer([2, 1], 64, 'Padding', 'same') % 卷积层，大小为 2x1，64个滤波器
    batchNormalizationLayer() % 批量归一化层
    reluLayer() % 激活函数 ReLU
    dropoutLayer(0.2) % Dropout层，丢弃20%的神经元
    
    fullyConnectedLayer(8) % 全连接层，8个神经元（修改为8个神经元，以匹配类别数）
    softmaxLayer() % Softmax 层
    classificationLayer() % 分类层
];

%% 训练模型
options = trainingOptions('sgdm', 'MaxEpochs', 10, 'InitialLearnRate', 0.01, 'Verbose', false);
net = trainNetwork(p_train, t_train, layers, options);

%% 预测模型
t_sim1 = predict(net, p_train);  % 训练集预测
t_sim_val = predict(net, p_val); % 验证集预测
t_sim2 = predict(net, p_test);   % 测试集预测

%% 将预测结果转换为分类标签
T_sim1 = categorical(vec2ind(t_sim1'))';   % 训练集预测结果
T_sim_val = categorical(vec2ind(t_sim_val'))';  % 验证集预测结果
T_sim2 = categorical(vec2ind(t_sim2'))';   % 测试集预测结果

%% 性能评价
% 将 T_train 转换为 categorical 类型
T_train = categorical(T_train);
T_test = categorical(T_test);
t_val = categorical(t_val);

% 训练集准确率
error_train = sum(T_sim1 == T_train) / M * 100;  

% 验证集准确率
error_val = sum(T_sim_val == t_val) / V * 100;  

% 测试集准确率
error_test = sum(T_sim2 == T_test) / N * 100;  

fprintf('训练集准确率：%.2f%%\n', error_train);
fprintf('验证集准确率：%.2f%%\n', error_val);
fprintf('测试集准确率：%.2f%%\n', error_test);

%% 绘制网络分析图
analyzeNetwork(layers)

%% 绘制混淆矩阵
% 训练集混淆矩阵
figure
cm_train = confusionchart(T_train, T_sim1);
cm_train.Title = 'Confusion Matrix for Train Data';
cm_train.ColumnSummary = 'column-normalized';
cm_train.RowSummary = 'row-normalized';

% 验证集混淆矩阵
figure
cm_val = confusionchart(t_val, T_sim_val);
cm_val.Title = 'Confusion Matrix for Validation Data';
cm_val.ColumnSummary = 'column-normalized';
cm_val.RowSummary = 'row-normalized';

% 测试集混淆矩阵
figure
cm_test = confusionchart(T_test, T_sim2);
cm_test.Title = 'Confusion Matrix for Test Data';
cm_test.ColumnSummary = 'column-normalized';
cm_test.RowSummary = 'row-normalized';

%% 计算精确率、召回率和 F1 分数
% 计算混淆矩阵中的每个元素
[precision_train, recall_train, f1_train] = calculateMetrics(cm_train);
[precision_val, recall_val, f1_val] = calculateMetrics(cm_val);
[precision_test, recall_test, f1_test] = calculateMetrics(cm_test);

% 显示精确率、召回率和 F1 分数
fprintf('训练集精确率：%.2f%%\n', precision_train * 100);
fprintf('训练集召回率：%.2f%%\n', recall_train * 100);
fprintf('训练集F1分数：%.2f\n', f1_train);

fprintf('验证集精确率：%.2f%%\n', precision_val * 100);
fprintf('验证集召回率：%.2f%%\n', recall_val * 100);
fprintf('验证集F1分数：%.2f\n', f1_val);

fprintf('测试集精确率：%.2f%%\n', precision_test * 100);
fprintf('测试集召回率：%.2f%%\n', recall_test * 100);
fprintf('测试集F1分数：%.2f\n', f1_test);

%% 计算精确率、召回率和 F1 分数的函数
function [precision, recall, f1] = calculateMetrics(cm)
    % 从混淆矩阵提取各类的真阳性、假阳性、真阴性、假阴性
    TP = cm.NormalizedValues(1,1);
    FP = cm.NormalizedValues(1,2);
    FN = cm.NormalizedValues(2,1);
    TN = cm.NormalizedValues(2,2);
    
    % 计算精确率、召回率和 F1分数
    precision = TP / (TP + FP);  % 精确率 = TP / (TP + FP)
    recall = TP / (TP + FN);     % 召回率 = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall);  % F1分数 = 2 * (精确率 * 召回率) / (精确率 + 召回率)
end
