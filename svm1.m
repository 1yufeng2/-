%% 清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%% 添加路径
addpath('LSSVM_Toolbox\')

%% 导入数据
res = xlsread('8 phosphor data sets.xlsx');

%% 划分训练集、验证集和测试集
temp = randperm(960); % 生成乱序数组，样本数目

% 按比例 3:1:1 划分数据集
p_train = res(temp(1:576), 1:20)';          % 训练集 60%
t_train = res(temp(1:576), 21)';
p_val   = res(temp(577:768), 1:20)';        % 验证集 20%
t_val  = res(temp(577:768), 21)';
p_test  = res(temp(769:end), 1:20)';        % 测试集 20%
t_test  = res(temp(769:end), 21)';
M = size(p_train, 2);
V = size(p_val, 2);
N = size(p_test, 2);

%% 转置以适应模型
p_train = p_train'; p_val=p_val'; p_test = p_test';
t_train = t_train'; t_val = t_val'; t_test = t_test';

%% 参数设置
gam  = 10;                    % 核函数参数
sig2 = 0.5;                   % 惩罚参数
type = 'c';                   % 模型类型 分类
codefct = 'code_OneVsOne';    % 一对一编码（推荐）
%          code_OneVsAll      % 一对多编码
kernel_type = 'RBF_kernel';   % RBF 核函数  
%              poly_kernel    % 多项式核函数 
%              MLP_kernel     % 多层感知机核函数
%              lin_kernel     % 线性核函数

%% 编码
[t_train, codebook, old_codebook] = code(t_train, codefct);

%% 建立模型
model = initlssvm(p_train, t_train, type, gam, sig2, kernel_type, codefct); 

%% 训练模型
model = trainlssvm(model);

%% 测试模型
t_sim1 = simlssvm(model, p_train);  % 训练集预测
t_sim2 = simlssvm(model, p_test );  % 测试集预测
t_sim_val = simlssvm(model, p_val); % 验证集预测

%% 解码
T_sim1 = code(t_sim1, old_codebook, [], codebook);
T_sim2 = code(t_sim2, old_codebook, [], codebook);
T_sim_val = code(t_sim_val, old_codebook, [], codebook);

%% 数据排序
[T_train, index_1] = sort(t_train);
[T_val, index_2] = sort(t_val);  % 验证集的排序
[T_test, index_3] = sort(t_test);

% 按排序索引调整预测结果
T_sim1 = T_sim1(index_1);
T_sim_val = T_sim_val(index_2);
T_sim2 = T_sim2(index_3);

%% 性能评价
% 检查 T_sim1 和 T_train 是否具有相同的维度
if size(T_sim1, 1) ~= size(T_train, 1)
    T_sim1 = T_sim1'; % 如果 T_sim1 是行向量，则转置为列向量
end
if size(T_sim1, 1) ~= size(T_train, 1)
    error('T_sim1 和 T_train 的维度不匹配');
end

error1 = sum((T_sim1 == T_train)) / M * 100 ;  % 训练集准确率
error2 = sum((T_sim2 == T_test )) / N * 100 ;  % 测试集准确率
error_val = sum((T_sim_val == T_val)) / V * 100; % 验证集准确率

%% 计算精确率、召回率和F1分数
% 使用 confusionmat 函数计算混淆矩阵
conf_train = confusionmat(T_train, T_sim1);
conf_val = confusionmat(T_val, T_sim_val);
conf_test = confusionmat(T_test, T_sim2);

% 训练集
precision_train = conf_train(1,1) / (conf_train(1,1) + conf_train(1,2));
recall_train = conf_train(1,1) / (conf_train(1,1) + conf_train(2,1));
f1_train = 2 * (precision_train * recall_train) / (precision_train + recall_train);

% 验证集
precision_val = conf_val(1,1) / (conf_val(1,1) + conf_val(1,2));
recall_val = conf_val(1,1) / (conf_val(1,1) + conf_val(2,1));
f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val);

% 测试集
precision_test = conf_test(1,1) / (conf_test(1,1) + conf_test(1,2));
recall_test = conf_test(1,1) / (conf_test(1,1) + conf_test(2,1));
f1_test = 2 * (precision_test * recall_test) / (precision_test + recall_test);

% 输出结果
fprintf('训练集准确率：%.2f%%\n', error1);
fprintf('验证集准确率：%.2f%%\n', error_val);
fprintf('测试集准确率：%.2f%%\n', error2);

fprintf('训练集精确率：%.2f\n', precision_train);
fprintf('训练集召回率：%.2f\n', recall_train);
fprintf('训练集F1分数：%.2f\n', f1_train);

fprintf('验证集精确率：%.2f\n', precision_val);
fprintf('验证集召回率：%.2f\n', recall_val);
fprintf('验证集F1分数：%.2f\n', f1_val);

fprintf('测试集精确率：%.2f\n', precision_test);
fprintf('测试集召回率：%.2f\n', recall_test);
fprintf('测试集F1分数：%.2f\n', f1_test);

%% 绘图：训练集、验证集、测试集的预测结果对比
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'训练集预测结果对比'; ['准确率=' num2str(error1) '%']};
title(string)
grid

figure
plot(1: V, T_val, 'r-*', 1: V, T_sim_val, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'验证集预测结果对比'; ['准确率=' num2str(error_val) '%']};
title(string)
grid

figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'测试集预测结果对比'; ['准确率=' num2str(error2) '%']};
title(string)
grid

%%% 混淆矩阵：训练集、验证集、测试集的混淆矩阵
% 训练集混淆矩阵
figure
cm_train = confusionchart(T_train, T_sim1);
cm_train.Title = 'Confusion Matrix for Train Data';
cm_train.ColumnSummary = 'column-normalized';
cm_train.RowSummary = 'row-normalized';

% 验证集混淆矩阵
figure
cm_val = confusionchart(T_val, T_sim_val);
cm_val.Title = 'Confusion Matrix for Validation Data';
cm_val.ColumnSummary = 'column-normalized';
cm_val.RowSummary = 'row-normalized';

% 测试集混淆矩阵
figure
cm_test = confusionchart(T_test, T_sim2);
cm_test.Title = 'Confusion Matrix for Test Data';
cm_test.ColumnSummary = 'column-normalized';
cm_test.RowSummary = 'row-normalized';
