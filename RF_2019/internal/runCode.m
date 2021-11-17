%% Step 1 : Get BoW database from Kmeans 
tic 
[data_train, data_query] = getData_mine('Caltech');

%% Step 2: Moderating database from 3d matrix to 2d matrix

S = size(data_train);
data = [];
for i = 1:S(1)
    % Labels are added to the descriptors 
    newData = [reshape(data_train(i,:,:),[S(2),S(3)]) i*ones(S(2),1)];
    % Data matrix is expanded
    data = [data; newData];
end 

data_train2 = data; % moderated training dataset

S = size(data_query);
data = [];
for i = 1:S(1)
    % Labels are added to the descriptors 
    newData = [reshape(data_query(i,:,:),[S(2),S(3)]) i*ones(S(2),1)];
    % Data matrix is expanded
    data = [data; newData];
end
data_query2 = data; % moderated testing dataset

%% Step 3:  Growing classification trees and testing

disp('Training Random Forest')

param.num = 50;    % Number of trees
param.depth = 15;    % trees depth
param.splitNum = 10; % Number of trials in split function
param.split = 'IG'; % Currently support 'information gain' only

tree=growTrees(data_train2,param);

%% Part 4: Testing the classification trees

disp('Testing Random Forest')
label=testTrees_fast(data_query2(:,1:end-1),tree); % Not including labels

C=[]; % The class labels after classification
for i=1:150
    [~,idx]=max(sum(tree(1).prob(label(i,:),:)));
    C = [C idx];
end

conf = confusionmat(data_query2(:,end)',C); % Confusion matrix

toc

% calculate accuracy
A= C==data_query2(:,end)';
Correctness = 100*sum(A)/150; % Percentage Correctness
disp('Classification Accuracy (in percentage): ');
disp(Correctness);

% Confusion matrix
figure
image(conf*(100/15))
colorbar
title('Confusion Matrix for RF classifier')
xlabel('Actual Class')
ylabel('Predicted Class')
