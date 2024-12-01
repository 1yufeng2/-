 %%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%%  导入数据
res = xlsread('8 phosphor data sets.xlsx');

%%  划分训练集和测试集
temp = randperm(320);%%生成乱序数组，样本数目

P_train = res(temp(1:240), 1:20)';%%先行后列
T_train = res(temp(1:240), 21)';
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

%%  训练模型
trees = 40;                                       % 决策树数目
leaf  = 1;                                        % 最小叶子数
OOBPrediction = 'on';                             % 打开误差图
OOBPredictorImportance = 'on';                    % 计算特征重要性
Method = 'classification';                        % 分类还是回归
net = TreeBagger(trees, p_train, t_train, 'OOBPredictorImportance', OOBPredictorImportance, ...
      'Method', Method, 'OOBPrediction', OOBPrediction, 'minleaf', leaf);
importance = net.OOBPermutedPredictorDeltaError;  % 重要性

%%  仿真测试
t_sim1 = predict(net, p_train);
t_sim2 = predict(net, p_test );

%%  格式转换
T_sim1 = str2double(t_sim1);
T_sim2 = str2double(t_sim2);
 
%%  性能评价
error1 = sum((T_sim1' == T_train)) / M * 100 ;
error2 = sum((T_sim2' == T_test )) / N * 100 ;

%%  绘制误差曲线
figure
plot(1: trees, oobError(net), 'b-', 'LineWidth', 1)
legend('Error curve','FontName','Time New Roman')
xlabel('Decision tree number','FontName','Time New Roman')
ylabel('Error','FontName','Time New Roman')
xlim([1, trees])
grid

%%  绘制特征重要性
figure
bar(importance)
legend('Importance','FontName','Time New Roman')
xlabel('Feature','FontName','Time New Roman')
ylabel('Importance','FontName','Time New Roman')

%%  数据排序
[T_train, index_1] = sort(T_train);
[T_test , index_2] = sort(T_test );

T_sim1 = T_sim1(index_1);
T_sim2 = T_sim2(index_2);

%%  绘图
figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('True value', 'Predicted value','FontName','Time New Roman')
xlabel('Predictive sample','FontName','Time New Roman')
ylabel('Prediction result','FontName','Time New Roman')
string = {'Comparison of prediction results of training set'; ['Accuracy rate=' num2str(error1) '%']};
title(string)
grid
set(gca,'Fontname','Time New Roman');
figure
plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
legend('True value', 'Predicted value','FontName','Time New Roman')
xlabel('Predictive sample','FontName','Time New Roman')
ylabel('Prediction result','FontName','Time New Roman')
string = {'Comparison of prediction results of Test set'; ['Accuracy rate=' num2str(error2) '%']};
title(string)
grid
set(gca,'Fontname','Time New Roman');
%%  混淆矩阵
figure
cm = confusionchart(T_train, T_sim1);
cm.Title = 'Confusion Matrix for Train Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
set(gca,'Fontname','Time New Roman');    
figure
cm = confusionchart(T_test, T_sim2);
cm.Title = 'Confusion Matrix for Test Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
set(gca,'Fontname','Time New Roman');