%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%%  添加路径
addpath('LSSVM_Toolbox\')

%%  导入数据
res = xlsread('8 phosphor data sets.xlsx');

%%  划分训练集和测试集
temp = randperm(320);

P_train = res(temp(1: 240), 1: 20)';
T_train = res(temp(1: 240), 21)';
M = size(P_train, 2);

P_test = res(temp(241: end), 1: 20)';
T_test = res(temp(241: end), 21)';
N = size(P_test, 2);

%%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test = mapminmax('apply', P_test, ps_input );
t_train = T_train;
t_test  = T_test ;

%%  转置以适应模型
p_train = p_train'; p_test = p_test';
t_train = t_train'; t_test = t_test';

%%  参数设置
gam  = 10;                    % 核函数参数
sig2 = 0.5;                   % 惩罚参数
type = 'c';                   % 模型类型 分类
codefct = 'code_OneVsOne';    % 一对一编码（推荐）
%          code_OneVsAll      % 一对多编码
kernel_type = 'RBF_kernel';   % RBF 核函数  
%              poly_kernel    % 多项式核函数 
%              MLP_kernel     % 多层感知机核函数
%              lin_kernel     % 线性核函数

%%  编码
[t_train, codebook, old_codebook] = code(t_train, codefct);

%%  建立模型
model = initlssvm(p_train, t_train, type, gam, sig2, kernel_type, codefct); 

%%  训练模型
model = trainlssvm(model);

%%  测试模型
t_sim1 = simlssvm(model, p_train);
t_sim2 = simlssvm(model, p_test ); 

%%  解码
T_sim1 = code(t_sim1, old_codebook, [], codebook);
T_sim2 = code(t_sim2, old_codebook, [], codebook);

%%  数据排序
[T_train, index_1] = sort(T_train);
[T_test , index_2] = sort(T_test );

T_sim1 = T_sim1(index_1);
T_sim2 = T_sim2(index_2);

%%  性能评价
error1 = sum((T_sim1' == T_train)) / M * 100 ;
error2 = sum((T_sim2' == T_test )) / N * 100 ;

%%  绘图
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('真实值', '预测值')
xlabel('预测样本')
ylabel('预测结果')
string = {'训练集预测结果对比'; ['准确率=' num2str(error1) '%']};
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

%%  混淆矩阵
figure
cm = confusionchart(T_train, T_sim1);
cm.Title = 'Confusion Matrix for Train Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
    
figure
cm = confusionchart(T_test, T_sim2);
cm.Title = 'Confusion Matrix for Test Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';