clear; clc; close all;
T = 0.01;
runTime = 200;
t = 0:T:runTime;
nState = 6;
nMeausre = 3;
initPos = [15; 15; 200];
initVel = [7; 7; 0];
totalScans = length(t);
R = [9 0 0;
    0 1 0;
    0 0 0.1];
x0(:,1) = [initPos; initVel] + [randn(3,1); 16*randn(3,1)]; %initial estimated state
P0 = [9*eye(3,3) zeros(3,3); zeros(3,3), 16^2*eye(3,3)];
noise = sqrt(R)*randn(3, totalScans);
xtrue = zeros(nState, totalScans);
xest = zeros(nState, totalScans);
ytrue = zeros(nMeausre, totalScans);
yest = zeros(nMeausre, totalScans);
ynoise = zeros(nMeausre, totalScans);
xtrue(:,1) = [initPos; initVel];                            %initial true state
% fState = @(xState)[xState(1)+ T*xState(4); xState(2)+T*xState(5);xState(3)+T*xState(6);xState(4); xState(5); xState(6)];
fState = @(xState)[xState(1:3)+ T*xState(4:6);xState(4:6)];
yFunc = @(state)[sqrt(state(1)^2 + state(2)^2 + state(3)^2); 
        atan2d(state(2),state(1));
        atan2d(state(3),(sqrt(state(1)^2 +state(2)^2)))];
h(:,1)=[201,45,54];
%calculate the true state
for i = 2:totalScans
    xtrue(:,i) = feval(fState,xtrue(:,i-1));
end 

%calculate measurements
for i = 1:totalScans
    ytrue(:,i) = [sqrt(xtrue(1,i)^2 + xtrue(2,i)^2 + xtrue(3,i)^2);
                atan2d(xtrue(2,i), xtrue(1,i));
                atan2d(xtrue(3,i), sqrt(xtrue(1,i)^2 + xtrue(2,i)^2 ))];
    ynoise(:,i) = ytrue(:,i) + noise(:,i);
end

% EKF
for k = 1:totalScans
    if k == 1
        xest(:,1) = x0(:,1);
        P = P0;
    else
        F = numeric_jacobian(fState, xest(:,k-1));
        xpred = feval(fState,xest(:,k-1));
        Ppred = F*P*F';
        x = xpred(1);
        y = xpred(2);
        z = xpred(3);
        xM = [x y z 0 0 0];
        yest(:,k) = feval( yFunc, xM);
        H = numeric_jacobian(yFunc,xM);
        S = (H*Ppred*H' + R);
        K = Ppred*H'*inv(S);
        yk = ynoise(:,k) - yest(:,k);
        xest(:,k) = xpred + K*yk;
        P = (eye(nState) - K*H)*Ppred;
        h(:,k)=[sqrt(xest(1,k)^2 + xest(2,k)^2 + xest(3,k)^2); ...
                atan2d(xest(2,k),xest(1,k)); ...
                atan2d(xest(3,k),(sqrt(xest(1,k)^2 +xest(2,k)^2)))];
    end
end
%% plot
figure
subplot(2,2,1)
plot(t,xtrue(1,:),'g','LineWidth',1.5)
hold on
plot(t,xest(1,:),'r--','LineWidth',1.5)
legend( 'True Position on x axis', 'Estimated Position on x axis')
xlabel('Time (s)')
ylabel('position (m)')
title('Target Position on x axis ')
grid on

subplot(2,2,2) 
plot(t,xtrue(2,:),'g','LineWidth',1.5)
hold on
plot(t,xest(2,:),'r--','LineWidth',1.5)
legend( 'True Position on y axis', 'Estimated Position on y axis')
xlabel('Time (s)')
ylabel('position (m)')
title('Target Position on y axis ')
grid on

subplot(2,2,3) 
plot(t,xtrue(3,:),'g','LineWidth',1.5)
hold on
plot(t,xest(3,:),'r--','LineWidth',1.5)
legend( 'True Position on z axis', 'Estimated Position on z axis')
xlabel('Time (s)')
ylabel('position (m)')
title('Target Position on z axis ')
grid on
%% function
function jac = numeric_jacobian(f, x)
    % Calculate Jacobian of function f at given x
    epsilon = 1e-6;
    epsilon_inv = 1/epsilon;
    nx = length(x); % Dimension of the input x;
    f0 = feval(f, x); % caclulate f0, when no perturbation happens
    % Do perturbation
    for i = 1 : nx
        x_ = x;
        x_(i) =  x(i) + epsilon;
        jac(:, i) = (feval(f, x_) - f0) .* epsilon_inv;
    end
end
% F=[1 0 0 T 0 0;          % Transition matrix
%     0 1 0 0 T 0;
%     0 0 1 0 0 T;
%     0 0 0 1 0 0;
%     0 0 0 0 1 0;
%     0 0 0 0 0 1];

%  H=[x/(x^2 + y^2 + z^2)^(1/2),                        y/(x^2 + y^2 + z^2)^(1/2),                   z/(x^2 + y^2 + z^2)^(1/2), 0, 0, 0;
%   -y/(x^2*(y^2/x^2 + 1)),                              1/(x*(y^2/x^2 + 1)),                                           0, 0, 0, 0;
%   -(x*z)/((z^2/(x^2 + y^2 + z^2) + 1)*(x^2 + y^2 + z^2)^(3/2)), -(y*z)/((z^2/(x^2 + y^2 + z^2) + 1)*(x^2 + y^2 + z^2)^(3/2)), (1/(x^2 + y^2 + z^2)^(1/2) - z^2/(x^2 + y^2 + z^2)^(3/2))/(z^2/(x^2 + y^2 + z^2) + 1), 0, 0, 0];
