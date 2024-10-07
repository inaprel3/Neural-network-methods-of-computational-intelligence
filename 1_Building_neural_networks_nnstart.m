% Створення навчальної вибірки для регресії (лінійна функція)
X = (1:100)'; % Вхідні дані: вектори значень від 1 до 100
y = 2*X + 5 + randn(size(X)); % Вихідні дані: лінійна функція y = 2x + 5 + шум

figure;
plot(X, y, 'bo');
title('Лінійна регресія з шумом');
xlabel('x');
ylabel('y');

nnstart

net1 = feedforwardnet(5); % 5 нейронів у прихованому шарі
net2 = feedforwardnet(10); % 10 нейронів у прихованому шарі
net3 = feedforwardnet(15);  % 15 нейронів у прихованому шарі

net1 = train(net1, X', y'); % Навчання мережі з 5 нейронами
net2 = train(net2, X', y'); % Навчання мережі з 10 нейронами
net3 = train(net3, X', y'); % Навчання мережі з 15 нейронами

predictions1 = net1(X');
mse1 = perform(net1, y', predictions1);
predictions2 = net2(X');
mse2 = perform(net2, y', predictions2);
predictions3 = net3(X');
mse3 = perform(net3, y', predictions3);

[mse_best, idx] = min([mse1, mse2, mse3]); % Пошук мінімального MSE

fprintf('MSE для першої мережі (5 нейронів): %.4f\n', mse1);
MSE для першої мережі (5 нейронів): 0.7427
fprintf('MSE для другої мережі (10 нейронів): %.4f\n', mse2);
MSE для другої мережі (10 нейронів): 1.0403
fprintf('MSE для третьої мережі (15 нейронів): %.4f\n', mse3);
MSE для третьої мережі (15 нейронів): 0.9824
fprintf('Найкраща архітектура — мережа з %d нейронами в прихованому шарі\n', idx * 5);
Найкраща архітектура — мережа з 5 нейронами в прихованому шарі
