nntool

% Вхідні дані
X = (1:100); 

% Вихідні дані (лінійна залежність з шумом)
y = 2*X + 5 + randn(size(X));

% Архітектура 1: 2 прихованих шари, [10, 5] нейронів
net1 = feedforwardnet([10, 5]);

% Архітектура 2: 3 прихованих шари, [15, 10, 5] нейронів
net2 = feedforwardnet([15, 10, 5]);

% Архітектура 3: 3 прихованих шари, [20, 15, 10] нейронів
net3 = feedforwardnet([20, 15, 10]);

% Налаштування пропорцій навчальних, валідаційних та тестових даних для всіх мереж
net1.divideParam.trainRatio = 0.7; % 70% для навчання
net1.divideParam.valRatio = 0.15;  % 15% для валідації
net1.divideParam.testRatio = 0.15; % 15% для тестування

net2.divideParam.trainRatio = 0.7;
net2.divideParam.valRatio = 0.15;
net2.divideParam.testRatio = 0.15;

net3.divideParam.trainRatio = 0.7;
net3.divideParam.valRatio = 0.15;
net3.divideParam.testRatio = 0.15;

% Навчання першої мережі (2 прихованих шари, [10, 5] нейронів)
[net1, tr1] = train(net1, X, y);

% Навчання другої мережі (3 прихованих шари, [15, 10, 5] нейронів)
[net2, tr2] = train(net2, X, y);

% Навчання третьої мережі (3 прихованих шари, [20, 15, 10] нейронів)
[net3, tr3] = train(net3, X, y);

% Вихідні дані для кожної мережі
outputs1 = net1(X);
outputs2 = net2(X);
outputs3 = net3(X);

% Обчислення помилки для кожної мережі
errors1 = gsubtract(y, outputs1); % Помилка для мережі 1
errors2 = gsubtract(y, outputs2); % Помилка для мережі 2
errors3 = gsubtract(y, outputs3); % Помилка для мережі 3

% Обчислення середньоквадратичної помилки (MSE)
mse1 = mean(errors1.^2);
mse2 = mean(errors2.^2);
mse3 = mean(errors3.^2);

% Вибір найкращої мережі на основі MSE
[min_mse, best_net_idx] = min([mse1, mse2, mse3]);
fprintf('Найкраща мережа - №%d з MSE = %.4f\n', best_net_idx, min_mse);
Найкраща мережа - №1 з MSE = 0.8297
 
% Двошарова мережа з Практичної роботи №1
net_practical1 = feedforwardnet(10); % Наприклад, 10 нейронів у прихованому шарі
[net_practical1, tr_practical1] = train(net_practical1, X, y);

% Вихідні дані та помилка для двошарової мережі
outputs_practical1 = net_practical1(X);
errors_practical1 = gsubtract(y, outputs_practical1);
mse_practical1 = mean(errors_practical1.^2);

% Порівняння з результатами багатошарових мереж
fprintf('MSE для двошарової мережі (Практична 1): %.4f\n', mse_practical1);
fprintf('MSE для найкращої багатошарової мережі: %.4f\n', min_mse);
MSE для двошарової мережі (Практична 1): 2.6638
MSE для найкращої багатошарової мережі: 0.8297
 
% Кількість параметрів для двошарової мережі
num_params_practical1 = sum(sum(net_practical1.IW{1})) + sum(sum(net_practical1.LW{2})) + length(net_practical1.b{1}) + length(net_practical1.b{2});

% Кількість параметрів для кожної з багатошарових мереж
num_params1 = sum(sum(net1.IW{1})) + sum(sum(net1.LW{2})) + length(net1.b{1}) + length(net1.b{2});
num_params2 = sum(sum(net2.IW{1})) + sum(sum(net2.LW{2})) + length(net2.b{1}) + length(net2.b{2});
num_params3 = sum(sum(net3.IW{1})) + sum(sum(net3.LW{2})) + length(net3.b{1}) + length(net3.b{2});

% Виведення результатів
fprintf('Кількість параметрів у мережі з Практичної 1: %d\n', num_params_practical1);
Кількість параметрів у мережі з Практичної 1: 1.039809e+02

fprintf('Кількість параметрів у мережі 1: %d\n', num_params1);
Кількість параметрів у мережі 1: 1.077056e+01
fprintf('Кількість параметрів у мережі 2: %d\n', num_params2);
Кількість параметрів у мережі 2: 4.176370e+00
fprintf('Кількість параметрів у мережі 3: %d\n', num_params3);
Кількість параметрів у мережі 3: 7.852101e+01
