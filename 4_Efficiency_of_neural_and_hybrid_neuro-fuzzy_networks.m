% Визначаємо кількість зразків
n = 100;

% Генеруємо випадкові значення для вхідних змінних X1, X2, X3
X1 = rand(n, 1) * 10;  % випадкові значення від 0 до 10
X2 = rand(n, 1) * 20;  % випадкові значення від 0 до 20
X3 = rand(n, 1) * 30;  % випадкові значення від 0 до 30

% Задаємо вихідну змінну Y зі штучною залежністю від X1, X2 та X3
% Наприклад, як лінійна комбінація з додаванням шуму
Y = 2 * X1 + 3 * X2 - X3 + randn(n, 1) * 5;  % додаємо шум для реалістичності

% Об’єднуємо всі дані в одну таблицю для зручності
data = table(X1, X2, X3, Y);

% Зберігаємо таблицю 'data' у файл 'training_data.mat'
save('training_data.mat', 'data');

% Завантаження даних
load('training_data.mat');  % Завантажуємо таблицю 'data'

% Розділення на вхідні та вихідні змінні
X = [data.X1, data.X2, data.X3]';  % Вхідні змінні (матриця 3xN)
Y = data.Y';                        % Вихідна змінна (1xN)

% Визначення параметрів навчання
hiddenLayerSizes = [5, 10, 15, 20, 25];  % Кількість нейронів для експериментів
results = [];  % Масив для збереження MSE для кожної кількості нейронів

for i = 1:length(hiddenLayerSizes)
    % Створення двошарової нейронної мережі
    net = feedforwardnet(hiddenLayerSizes(i));  % Налаштовуємо прихований шар

% Розділення даних для навчання, валідації та тестування
    net.divideParam.trainRatio = 0.7;  % 70% на навчання
    net.divideParam.valRatio = 0.15;   % 15% на валідацію
    net.divideParam.testRatio = 0.15;  % 15% на тестування
    
    % Навчання мережі
    net = train(net, X, Y);

% Перевірка точності моделі
    predictions = net(X);             % Прогнози для навчальної вибірки
    mseError = perform(net, Y, predictions);  % Розрахунок MSE
    
    % Збереження результатів
    results = [results; hiddenLayerSizes(i), mseError];  % Додаємо до результатів
    
    fprintf('Hidden neurons: %d, MSE: %.4f\n', hiddenLayerSizes(i), mseError);
end

% Вибір найкращого варіанту
[~, bestIndex] = min(results(:, 2));
bestHiddenSize = results(bestIndex, 1);
fprintf('Найкраща модель має %d нейронів у прихованому шарі з MSE: %.4f\n', bestHiddenSize, results(bestIndex, 2));

% Завантаження даних
load('training_data.mat');  % Завантажуємо таблицю 'data'

% Розділення на вхідні та вихідні змінні
X = [data.X1, data.X2, data.X3]';  % Вхідні змінні (матриця 3xN)
Y = data.Y';                        % Вихідна змінна (1xN)

% Визначення конфігурацій мереж
% Кожен елемент масиву - кількість нейронів у кожному шарі для кожної конфігурації
networkConfigurations = {
    [10, 5],        % 2 шари: 10 нейронів у першому шарі, 5 - у другому
    [15, 10, 5],    % 3 шари: 15, 10 і 5 нейронів відповідно
    [20, 15, 10],   % 3 шари: 20, 15 і 10 нейронів відповідно
    [25, 20, 15, 10] % 4 шари: 25, 20, 15 і 10 нейронів відповідно
};

results = [];  % Масив для збереження MSE для кожної конфігурації

% Навчання мережі для кожної конфігурації
for i = 1:length(networkConfigurations)
    % Створення багатошарової нейронної мережі з поточною конфігурацією
    net = feedforwardnet(networkConfigurations{i});
    
    % Налаштування розділення даних для навчання, валідації та тестування
    net.divideParam.trainRatio = 0.7;  % 70% на навчання
    net.divideParam.valRatio = 0.15;   % 15% на валідацію
    net.divideParam.testRatio = 0.15;  % 15% на тестування
    
    % Навчання мережі
    net = train(net, X, Y);
    
    % Перевірка точності моделі
    predictions = net(X);             % Прогнози для навчальної вибірки
    mseError = perform(net, Y, predictions);  % Розрахунок MSE
    
    % Збереження результатів
    results = [results; {networkConfigurations{i}}, mseError];  % Додаємо до результатів
    
    % Виведення інформації про поточну конфігурацію
    fprintf('Configuration: %s, MSE: %.4f\n', mat2str(networkConfigurations{i}), mseError);
end

% Вибір найкращої конфігурації
[~, bestIndex] = min(cell2mat(results(:, 2)));
bestConfiguration = results{bestIndex, 1};
bestMSE = results{bestIndex, 2};
fprintf('Найкраща модель має конфігурацію %s з MSE: %.4f\n', mat2str(bestConfiguration), bestMSE);

load('training_data.mat');  % Завантажуємо таблицю 'data'
X = [data.X1, data.X2, data.X3];  % Вхідні змінні (матриця 3xN)
Y = data.Y;                       % Вихідна змінна (вектор N)

data = [X, Y];  % Об'єднуємо вхідні та вихідні дані

numTerms = [2, 3, 4];  % Кількість лінгвістичних термів

results = [];  % Масив для збереження MSE для кожної конфігурації

for terms = numTerms
    % Визначення правил та типів функцій належності
    mf = cell(3, 1);  % Масив для зберігання функцій належності
    mf{1} = 'trimf';  % Трикутна функція
    mf{2} = 'trapmf'; % Трапецієподібна функція
    mf{3} = 'gbellmf'; % Гаусова функція
    
    for j = 1:length(mf)
        % Налаштування ANFIS
        fis = genfis1(data, terms, mf{j});
        
        % Навчання ANFIS
        options = anfisOptions('InitialFIS', fis, 'EpochNumber', 100);
        [trainedFIS, trainError] = anfis(data, options);
        
        % Перевірка точності моделі
        predictions = evalfis(data(:, 1:3), trainedFIS);  % Прогнози
        mseError = mean((data(:, 4) - predictions).^2);  % Розрахунок MSE
        
        % Збереження результатів
        results = [results; terms, j, mseError];  % Кількість термів, тип функції, MSE
        fprintf('Terms: %d, MF Type: %d, MSE: %.4f\n', terms, j, mseError);
    end
end

[~, bestIndex] = min(results(:, 3));
bestTerms = results(bestIndex, 1);
bestMFType = results(bestIndex, 2);
bestMSE = results(bestIndex, 3);
fprintf('Найкраща конфігурація: %d термів, тип MF: %d з MSE: %.4f\n', bestTerms, bestMFType, bestMSE);

% Визначаємо кількість зразків
n = 100;

% Генеруємо випадкові значення для вхідних змінних X1, X2, X3
X1 = rand(n, 1) * 10;  % випадкові значення від 0 до 10
X2 = rand(n, 1) * 20;  % випадкові значення від 0 до 20
X3 = rand(n, 1) * 30;  % випадкові значення від 0 до 30

% Задаємо вихідну змінну Y зі штучною залежністю від X1, X2 та X3
% Наприклад, як лінійна комбінація з додаванням шуму
Y = 2 * X1 + 3 * X2 - X3 + randn(n, 1) * 5;  % додаємо шум для реалістичності

% Об'єднуємо вхідні та вихідні дані в одну матрицю
dataMatrix = [X1, X2, X3, Y];  % Результуюча матриця

% Зберігаємо дані у форматі .dat
dlmwrite('training_data.dat', dataMatrix, 'delimiter', '\t', 'precision', '%.6f');  % Використовуємо табуляцію як роздільник

disp('Дані збережено у файлі training_data.dat');

anfisedit

% results - масив, що містить кількість термів, тип MF та MSE
[~, bestIndex] = min(results(:, 3));  % Знайти індекс з найменшим MSE
bestTerms = results(bestIndex, 1);     % Найкраща кількість термів
bestMFType = results(bestIndex, 2);    % Найкращий тип MF
bestMSE = results(bestIndex, 3);        % Найкраще значення MSE

fprintf('Найкраща конфігурація: %d термів, тип MF: %d з MSE: %.4f\n', bestTerms, bestMFType, bestMSE);

figure;
bar(results(:, 3));  % Створення стовпчикової діаграми для MSE
xticks(1:size(results, 1));  % Встановлення міток по осі X
xticklabels(arrayfun(@(x) sprintf('Terms: %d, MF Type: %d', results(x, 1), results(x, 2)), 1:size(results, 1), 'UniformOutput', false));
xlabel('Конфігурації');
ylabel('MSE');
title('Порівняння MSE для різних конфігурацій');
grid on;

% Визначаємо кількість зразків
n = 100;

% Генеруємо випадкові значення для вхідних змінних X1, X2, X3
X1 = rand(n, 1) * 10;  % випадкові значення від 0 до 10
X2 = rand(n, 1) * 20;  % випадкові значення від 0 до 20
X3 = rand(n, 1) * 30;  % випадкові значення від 0 до 30

% Задаємо вихідну змінну Y зі штучною залежністю від X1, X2 та X3
Y = 2 * X1 + 3 * X2 - X3 + randn(n, 1) * 5;  % додаємо шум для реалістичності

% Об’єднуємо всі дані в одну таблицю для зручності
data = table(X1, X2, X3, Y);

% Зберігаємо таблицю 'data' у файл 'training_data.mat'
save('training_data.mat', 'data');

% Завантаження даних
load('training_data.mat');  % Завантажуємо таблицю 'data'

% Розділення на вхідні та вихідні змінні
X = [data.X1, data.X2, data.X3]';  % Вхідні змінні (матриця 3xN)
Y = data.Y';                        % Вихідна змінна (1xN)

% Визначення параметрів навчання
hiddenLayerSizes = [5, 10, 15];  % Кількість нейронів для експериментів
results = [];  % Масив для збереження MSE для кожної кількості нейронів
computationalCosts = []; % Масив для збереження обчислювальних витрат

for i = 1:length(hiddenLayerSizes)
    % Створення двошарової нейронної мережі
    net = feedforwardnet(hiddenLayerSizes(i));  % Налаштовуємо прихований шар

    % Розділення даних для навчання, валідації та тестування
    net.divideParam.trainRatio = 0.7;  % 70% на навчання
    net.divideParam.valRatio = 0.15;   % 15% на валідацію
    net.divideParam.testRatio = 0.15;  % 15% на тестування

    % Навчання мережі та вимірювання часу
    tic; % Початок вимірювання часу
    net = train(net, X, Y);
    trainingTime = toc; % Час навчання

    % Перевірка точності моделі
    predictions = net(X);             % Прогнози для навчальної вибірки
    mseError = perform(net, Y, predictions);  % Розрахунок MSE
    
    % Збереження результатів
    results = [results; hiddenLayerSizes(i), mseError];  % Додаємо до результатів
    computationalCosts = [computationalCosts; hiddenLayerSizes(i), net.numWeightElements, trainingTime]; % Додаємо до витрат

    fprintf('Hidden neurons: %d, MSE: %.4f, Training Time: %.4f seconds, Number of parameters: %d\n', ...
        hiddenLayerSizes(i), mseError, trainingTime, net.numWeightElements);
end

% Вибір найкращого варіанту
[~, bestIndex] = min(results(:, 2));
bestHiddenSize = results(bestIndex, 1);
fprintf('Найкраща модель має %d нейронів у прихованому шарі з MSE: %.4f\n', bestHiddenSize, results(bestIndex, 2));

% Порівняння трьох мереж за точністю та обчислювальними витратами
fprintf('\nПорівняння трьох мереж:\n');
fprintf('%-15s %-15s %-20s %-15s\n', 'Нейронів', 'MSE', 'Час навчання (с)', 'Кількість параметрів');
fprintf('------------------------------------------------------\n');
for i = 1:length(hiddenLayerSizes)
    fprintf('%-15d %-15.4f %-20.4f %-15d\n', ...
        results(i, 1), results(i, 2), computationalCosts(i, 3), computationalCosts(i, 2));
end

% Вибір найкращої конфігурації
[~, bestIndex] = min(results(:, 2));
bestHiddenSize = results(bestIndex, 1);
fprintf('Найкраща модель має %d нейронів у прихованому шарі з MSE: %.4f\n', bestHiddenSize, results(bestIndex, 2));
