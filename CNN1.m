%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%%  导入数据
res =xlsread('8种荧光粉2.xlsx');

%%  划分训练集和测试集
temp = randperm(320);%%生成乱序数组，样本数目

P_train = res(temp(1: 240), 1: 20)';%%先行后列
T_train = res(temp(1: 240), 21)';
M = size(P_train, 2); 

P_test = res(temp(241: end), 1: 20)';
T_test = res(temp(241: end), 21)';
N = size(P_test, 2); 
%%  数据归一化
[P_train, ps_input] = mapminmax(P_train, 0, 1);
P_test  = mapminmax('apply', P_test, ps_input);

t_train =  categorical(T_train)';
t_test  =  categorical(T_test )';

%%  数据平铺
%   将数据平铺成1维数据只是一种处理方式
%   也可以平铺成2维数据，以及3维数据，需要修改对应模型结构
%   但是应该始终和输入层数据结构保持一致
p_train =  double(reshape(P_train, 20, 1, 1, M));
p_test  =  double(reshape(P_test , 20, 1, 1, N));

%%  构造网络结构
layers = [
 imageInputLayer([20, 1, 1])                                % 输入层
 
 convolution2dLayer([2, 1], 32, 'Padding', 'same')          % 卷积核大小为 2*1 生成16个卷积
 batchNormalizationLayer                                    % 批归一化层
 reluLayer                                                  % relu 激活层
 
 maxPooling2dLayer([2, 1], 'Stride', [2, 1])                % 最大池化层 大小为 2*1 步长为 [2, 1]

 convolution2dLayer([2, 1], 32, 'Padding', 'same')          % 卷积核大小为 2*1 生成32个卷积
 batchNormalizationLayer                                    % 批归一化层
 reluLayer                                                  % relu 激活层

 fullyConnectedLayer(8)                                    % 将输出大小改为8（类别数）
 softmaxLayer                                               % 损失函数层
 classificationLayer];                                      % 分类层

%%  参数设置
options = trainingOptions('adam', ...      % Adam 梯度下降算法
    'MaxEpochs', 500, ...                  % 最大训练次数 500
    'InitialLearnRate', 1e-3, ...          % 初始学习率为 0.001
    'L2Regularization', 1e-4, ...          % L2正则化参数
    'LearnRateSchedule', 'piecewise', ...  % 学习率下降
    'LearnRateDropFactor', 0.1, ...        % 学习率下降因子 0.1
    'LearnRateDropPeriod', 450, ...        % 经过450次训练后 学习率为 0.001 * 0.1
    'Shuffle', 'every-epoch', ...          % 每次训练打乱数据集
    'ValidationPatience', Inf, ...         % 关闭验证
    'Plots', 'training-progress', ...      % 画出曲线
    'Verbose', false);

%%  训练模型
net = trainNetwork(p_train, t_train, layers, options);

%%  预测模型
t_sim1 = predict(net, p_train); 
t_sim2 = predict(net, p_test ); 

%%  反归一化
T_sim1 = vec2ind(t_sim1');
T_sim2 = vec2ind(t_sim2');

%%  性能评价
error1 = sum((T_sim1 == T_train)) / M * 100 ;
error2 = sum((T_sim2 == T_test )) / N * 100 ;

%%  绘制网络分析图
analyzeNetwork(layers)

%%  数据排序
[T_train, index_1] = sort(T_train);
[T_test , index_2] = sort(T_test );

T_sim1 = T_sim1(index_1);
T_sim2 = T_sim2(index_2);

%%  绘图
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('True value', 'Predicted value')
xlabel('Predictive sample')
ylabel('Prediction result')
string = {'Comparison of prediction results of training set'; ['Accuracy rate=' num2str(error1) '%']};
title(string)
xlim([1, M])
grid
set(0,'defaultAxesFontName', '<fontname>');
figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('True value', 'Predicted value')
xlabel('Predictive sample')
ylabel('Prediction result')
string = {'Comparison of prediction results of test set'; ['Accuracy rate=' num2str(error2) '%']};
title(string)
xlim([1, N])
grid
set(0,'defaultAxesFontName', '<fontname>');
%%  混淆矩阵
figure
cm = confusionchart(T_train, T_sim1);
cm.Title = 'Confusion Matrix for Train Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
set(0,'defaultAxesFontName', '<fontname>');   
figure
cm = confusionchart(T_test, T_sim2);
cm.Title = 'Confusion Matrix for Test Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
set(0,'defaultAxesFontName', '<fontname>');