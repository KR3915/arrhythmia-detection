opts = delimitedTextImportOptions("NumVariables", 32);
opts.DataLines = [1, Inf];
opts.Delimiter = ",";
opts.VariableNamesLine = 0;
opts.VariableTypes = [{'string', 'char'}, repmat({'double'}, 1, 30)];
data = readtable("wdbc.data", opts);
diagnosis = categorical(data{:,2})
X = data{:,3:end}
Y = diagnosis
cv = cvpartition(Y, 'HoldOut', 0.2);  % 70 % train, 20 % test
Xtrain = X(training(cv), :);
Ytrain = Y(training(cv));
Xtest  = X(test(cv), :);
Ytest  = Y(test(cv));
hold off
model = fitcsvm(Xtrain, Ytrain);
Ypred = predict(model, Xtest);
accuracy = mean(Ypred == Ytest) * 100;
fprintf('Testovací přesnost: %.2f %%\n', accuracy);
confusionchart(Ytest, Ypred);
title('Confusion Matrix - Test Set');
