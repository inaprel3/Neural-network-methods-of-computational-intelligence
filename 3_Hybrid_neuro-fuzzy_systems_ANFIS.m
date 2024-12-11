simulink
fuzzy

% Генерація навчальної вибірки
x1_train = linspace(0, 10, 50)';  % 50 рівномірних точок для x1
x2_train = linspace(1, 10, 50)';  % 50 рівномірних точок для x2

[X1_train, X2_train] = meshgrid(x1_train, x2_train);  % Створення сітки
Y_train = X1_train(:).^2 + log10(X2_train(:));        % Обчислення y

% Формування навчальної вибірки у форматі [x1, x2, y]
train_data = [X1_train(:), X2_train(:), Y_train];
writematrix(train_data, 'train_data.dat', 'Delimiter', 'tab');  % Збереження у .dat

% Генерація тестової вибірки
x1_test = linspace(0.5, 9.5, 30)';  % 30 точок для x1 зі зміщенням
x2_test = linspace(1.5, 9.5, 30)';  % 30 точок для x2 зі зміщенням

[X1_test, X2_test] = meshgrid(x1_test, x2_test);
Y_test = X1_test(:).^2 + log10(X2_test(:));

% Формування тестової вибірки у форматі [x1, x2, y]
test_data = [X1_test(:), X2_test(:), Y_test];
writematrix(test_data, 'test_data.dat', 'Delimiter', 'tab');  % Збереження у .dat

% Генерація перевірочної вибірки
x1_val = 10 * rand(20, 1);  % 20 випадкових значень для x1 у [0, 10]
x2_val = 9 * rand(20, 1) + 1;  % 20 випадкових значень для x2 у [1, 10]

Y_val = x1_val.^2 + log10(x2_val);

% Формування перевірочної вибірки у форматі [x1, x2, y]
val_data = [x1_val, x2_val, Y_val];
writematrix(val_data, 'val_data.dat', 'Delimiter', 'tab');  % Збереження у .dat

anfisedit
fuzzyLogicDesigner

% Завантаження навчальної вибірки
train_data = load('train_data.dat');  
test_data = load('test_data.dat');  
val_data = load('val_data.dat');  

% Ініціалізація ANFIS з трьома термами на вхід
fis = genfis1(train_data, 3, 'trimf');

% Навчання моделі з використанням гібридного методу
[trained_fis, trainError, stepSize, chkFis, chkError] = ...
    anfis(train_data, fis, 100, [0, 0, 0, 0], val_data);

% Візуалізація помилки навчання та перевірки
figure;
plot(1:length(trainError), trainError, 'b-', 1:length(chkError), chkError, 'r-');
legend('Train Error', 'Check Error');
xlabel('Epochs');
ylabel('Error');
title('ANFIS Training and Validation Error');

% Завантаження вибірок
train_data = load('train_data.dat');  % Навчальна вибірка
test_data = load('test_data.dat');    % Тестова вибірка
val_data = load('val_data.dat');      % Перевірочна вибірка

% Отримання вхідних та вихідних даних
x1_train = train_data(:, 1);
x2_train = train_data(:, 2);
y_train = train_data(:, 3);

x1_test = test_data(:, 1);
x2_test = test_data(:, 2);
y_test = test_data(:, 3);

x1_val = val_data(:, 1);
x2_val = val_data(:, 2);
y_val = val_data(:, 3);

% Обчислення виходу ANFIS для навчальної, тестової та перевірочної вибірок
train_outputs = evalfis([x1_train, x2_train], trained_fis);
test_outputs = evalfis([x1_test, x2_test], trained_fis);
val_outputs = evalfis([x1_val, x2_val], trained_fis);

% Обчислення помилок
train_error = y_train - train_outputs;  % Помилка на навчальній вибірці
test_error = y_test - test_outputs;      % Помилка на тестовій вибірці
val_error = y_val - val_outputs;         % Помилка на перевірочній вибірці

% Середня абсолютна похибка
mae_train = mean(abs(train_error));
mae_test = mean(abs(test_error));
mae_val = mean(abs(val_error));

% Вивід результатів
fprintf('Середня абсолютна похибка на навчальній вибірці: %.4f\n', mae_train);
fprintf('Середня абсолютна похибка на тестовій вибірці: %.4f\n', mae_test);
fprintf('Середня абсолютна похибка на перевірочній вибірці: %.4f\n', mae_val);

% Візуалізація
figure;
subplot(3, 1, 1);
plot(y_train, 'b', 'DisplayName', 'Фактичні значення (Навчальна)');
hold on;
plot(train_outputs, 'r--', 'DisplayName', 'Передбачені значення (Навчальна)');
legend;
title('Навчальна вибірка');

subplot(3, 1, 2);
plot(y_test, 'b', 'DisplayName', 'Фактичні значення (Тестова)');
hold on;
plot(test_outputs, 'r--', 'DisplayName', 'Передбачені значення (Тестова)');
legend;
title('Тестова вибірка');

subplot(3, 1, 3);
plot(y_val, 'b', 'DisplayName', 'Фактичні значення (Перевірочна)');
hold on;
plot(val_outputs, 'r--', 'DisplayName', 'Передбачені значення (Перевірочна)');
legend;
title('Перевірочна вибірка');

% Зменшення кількості термів
num_terms_input1 = 2;  % Кількість лінгвістичних термів для x1
num_terms_input2 = 3;  % Кількість лінгвістичних термів для x2

% Створення структури ANFIS
% Припустимо, що train_data вже завантажено
anfis_input = train_data(:, 1:2);  % Вхідні дані
anfis_output = train_data(:, 3);    % Вихідні дані

% Перевірка розмірів
if size(anfis_input, 2) ~= 2 || size(anfis_output, 2) ~= 1
    error('Вхідні дані повинні мати розміри Nx2, а вихідні - Nx1');
end

% Налаштування функцій належності
mf_input1 = {'trimf', 'trimf'};  % Трикутна функція належності для x1
mf_input2 = {'trimf', 'trimf', 'trimf'};  % Трикутна функція належності для x2
mf_output = 'linear';  % Вихідна функція належності

% Генерація FIS
fis = genfis([anfis_input anfis_output], 'genfis', 'OutputMembershipFunctionType', mf_output);

% Додавання функцій належності для входів
fis.Inputs(1).MembershipFunctions = mf_input1;  % Задаємо функції належності для першого входу
fis.Inputs(2).MembershipFunctions = mf_input2;  % Задаємо функції належності для другого входу

% Навчання ANFIS
trained_fis = anfis([anfis_input anfis_output], fis, [100, 0.01, 0.9, 1, 0]);
