%% 清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%% 导入数据
res = xlsread('8 phosphor data sets.xlsx');

%% 划分训练集、验证集和测试集
temp = randperm(960); % 生成乱序数组，样本数目

% 按比例 3:1:1 划分数据集
P_train = res(temp(1:576), 1:20)';          % 训练集 60%
T_train = res(temp(1:576), 21)';
P_val   = res(temp(577:768), 1:20)';        % 验证集 20%
T_val   = res(temp(577:768), 21)';
P_test  = res(temp(769:end), 1:20)';        % 测试集 20%
T_test  = res(temp(769:end), 21)';

M = size(P_train, 2);
V = size(P_val, 2);
N = size(P_test, 2);

%% 数据归一化
[P_train, ps_input] = mapminmax(P_train, 0, 1);
P_val = mapminmax('apply', P_val, ps_input);
P_test = mapminmax('apply', P_test, ps_input);

t_train = T_train';
t_val   = T_val';
t_test  = T_test';

% 转置以适应模型
p_train = P_train';
p_val   = P_val';
p_test  = P_test';

%% 训练模型
trees = 40;                                       % 决策树数目
leaf  = 1;                                        % 最小叶子数
OOBPrediction = 'on';                             % 打开误差图
OOBPredictorImportance = 'on';                    % 计算特征重要性
Method = 'classification';                        % 分类还是回归
net = TreeBagger(trees, p_train, t_train, 'OOBPredictorImportance', OOBPredictorImportance, ...
      'Method', Method, 'OOBPrediction', OOBPrediction, 'minleaf', leaf);
importance = net.OOBPermutedPredictorDeltaError;  % 重要性

%% 仿真测试
t_sim1 = predict(net, p_train);
t_sim_val = predict(net, p_val);
t_sim2 = predict(net, p_test);

%% 格式转换
T_sim1 = str2double(t_sim1);
T_sim_val = str2double(t_sim_val);
T_sim2 = str2double(t_sim2);

%% 性能评价
accuracy_train = sum((T_sim1' == T_train)) / M * 100;
accuracy_val   = sum((T_sim_val' == T_val)) / V * 100;
accuracy_test  = sum((T_sim2' == T_test)) / N * 100;

fprintf('训练集准确率：%.2f%%\n', accuracy_train);
fprintf('验证集准确率：%.2f%%\n', accuracy_val);
fprintf('测试集准确率：%.2f%%\n', accuracy_test);

%% 精确度、召回率和 F1 分数计算函数
num_classes = length(unique(T_train)); % 分类类别数
[precision_train, recall_train, f1_train] = calculate_metrics(T_train, T_sim1, num_classes);
[precision_val, recall_val, f1_val] = calculate_metrics(T_val, T_sim_val, num_classes);
[precision_test, recall_test, f1_test] = calculate_metrics(T_test, T_sim2, num_classes);

% 平均指标
avg_precision_test = mean(precision_test);
avg_recall_test = mean(recall_test);
avg_f1_test = mean(f1_test);

fprintf('测试集平均精确度：%.2f%%\n', avg_precision_test * 100);
fprintf('测试集平均召回率：%.2f%%\n', avg_recall_test * 100);
fprintf('测试集平均 F1 分数：%.2f\n', avg_f1_test);

%% 绘制误差曲线
figure
plot(1:trees, oobError(net), 'b-', 'LineWidth', 1)
legend('Error curve', 'FontName', 'Time New Roman')
xlabel('Decision tree number', 'FontName', 'Time New Roman')
ylabel('Error', 'FontName', 'Time New Roman')
xlim([1, trees])
grid

%% 绘制特征重要性
figure
bar(importance)
legend('Importance', 'FontName', 'Time New Roman')
xlabel('Feature', 'FontName', 'Time New Roman')
ylabel('Importance', 'FontName', 'Time New Roman')

%% 混淆矩阵
figure
cm = confusionchart(T_test, T_sim2);
cm.Title = 'Confusion Matrix for Test Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
set(gca, 'Fontname', 'Time New Roman');

%% 函数：计算精确度、召回率和 F1 分数
function [precision, recall, f1_score] = calculate_metrics(true_labels, predicted_labels, num_classes)
    confusion_mat = confusionmat(true_labels, predicted_labels); % 混淆矩阵
    precision = zeros(1, num_classes);
    recall = zeros(1, num_classes);
    f1_score = zeros(1, num_classes);
    
    for i = 1:num_classes
        TP = confusion_mat(i, i);                 % True Positives
        FP = sum(confusion_mat(:, i)) - TP;       % False Positives
        FN = sum(confusion_mat(i, :)) - TP;       % False Negatives
        
        precision(i) = TP / (TP + FP + eps);      % 精确度
        recall(i) = TP / (TP + FN + eps);         % 召回率
        f1_score(i) = 2 * precision(i) * recall(i) / (precision(i) + recall(i) + eps); % F1 分数
    end
end
