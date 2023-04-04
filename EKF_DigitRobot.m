clear; clc;
%% read data
addpath('../Digit_data/0331/test01')
file1 =  'forwardKine.csv';
file2 =  'imuReading.csv';
file3 =  'jointPosFile.csv';
file4 =  'groundTruthFile.csv';
file5 =  'deltaTime.csv';
file6 =  'imuOrient.csv';
file7 =  'contactFile.csv';
file8 =  'baseOrient.csv';
file9 =  'trueFootFile.csv';
file10 = 'timerFile.csv';

leftFoot = readmatrix(file1,"Range",'A:C');
rightFoot = readmatrix(file1,"Range",'D:F');
world_leftFoot = readmatrix(file9,"Range",'A:C');
world_rightFoot = readmatrix(file9,"Range",'D:F');
iumReading = readmatrix(file2);
jointPos = readmatrix(file3);
truePos = readmatrix(file4,'Range','A:C');
trueVel =  readmatrix(file4,'Range','D:F');
dt = csvread(file5);
imuOrient = readmatrix(file6);
baseOrient = readmatrix(file8);
leftContact = readmatrix(file7,"Range",'A:A');
rightContact = readmatrix(file7,"Range",'B:B');
time = readmatrix(file10);
simLen = length(time);
g =[0; 0; -9.81];
%% data processing
torso2IMU = -pi/2;
R_torso2IMU = [cos(torso2IMU) 0 sin(torso2IMU);
               0              1 0;
              -sin(torso2IMU) 0 cos(torso2IMU)];
linAcc = (R_torso2IMU*iumReading(:,1:3)')';
angVel = (R_torso2IMU*iumReading(:,4:6)')';



%% construct filter
nState = 16;
nMeausre = 6;
v0 = [0; 0; 0];
p0 = [0; 0; 0];
q0 = [1; 0; 0; 0];
p0_lf = [0; 0; 0];
p0_rf = [0; 0; 0];
x0 = [v0; p0; q0; p0_lf; p0_rf];
P0 = 100*eye(nState); 
xest = zeros(simLen,nState);


R = eye(6);
Q_contact = 0.01*eye(3);
Q_noContact = 1e10*eye(3);
for i = 1:simLen
    if i == 1
        xest(1,:) = x0;
        P = P0;
    else
        angVel_update = angVel(i-1,:);
        linAcc_update = linAcc(i-1,:);
        dt_update = dt(i);
        fState = @(x)[x(1:3,1)+dt_update*x(4:6,1)+1/2*(dt_update^2*(qRotationM(x)*linAcc_update'+g));      %p
                    x(4:6,1)+dt_update*(qRotationM(x)*linAcc_update'+g);                                   %v
                    (quatmultiply(x(7:10,1)',rot2quat(dt_update*angVel_update')'))';          %q
%                     (quatmultiply(rot2quat(dt_update*angVel_update')',x(7:10,1)'))';            %q    
                    x(11:13,1);                                                                 %p_leftFoot
                    x(14:16,1)];                                                                %p_rightFoot
        leftFoot_update = leftFoot(i-1,:)'; 
        rightFoot_update = rightFoot(i-1,:)';
        ykFun = @(xM)[qRotationM(xM)'*(xM(11:13,1)-xM(1:3,1));
                     qRotationM(xM)'*(xM(14:16,1)-xM(1:3,1))];
        Q = 0.01*eye(16);
        if (rightContact(i-1)== 0)
            Q(14:16,14:16) = Q_noContact;
        elseif (rightContact(i-1)==1)
            Q(14:16,14:16) = Q_contact;
        end
        if (leftContact(i-1)==0)
            Q(11:13,11:13) = Q_noContact;
        elseif(leftContact(i-1)==1)
            Q(11:13,11:13) = Q_contact;
        end
        xpred = feval(fState,xest(i-1,:)');
        F = numeric_jacobian(fState,xest(i-1,:)');
        Ppred = F*P*F'+Q;
        yk = [leftFoot_update; rightFoot_update] - feval(ykFun,xpred);
        Ck = qRotationM(xpred);
        H = numeric_jacobian(ykFun,xpred);
        S = (H*Ppred*H' + R);
        K = Ppred*H'/(S);
        xest(i,:) = xpred + K*yk;
        %normalize quaternion
        
        P = (eye(nState) - K*H)*Ppred;
    end
end
% counter = 0;
% contactInfo(1,1) = 1;
% contactInfo(1,2) = leftContact(1);
% for j = 2:simLen
%     if leftContact(j) ~= leftContact(j-1)
%         counter = counter + 1;
%         contactInfo(counter,1) = time(j);
%         contactInfo(counter,2) = leftContact(j);
%     end
%     
% end
%% eulor angles
eul_est = quat2eul(xest(:,7:10));
eul_true = quat2eul(baseOrient);
%% plot
close all
figure 
subplot(2,2,1)
plot(time,trueVel(:,1),time,xest(:,4))
xlabel('time (s)')
ylabel('x velocity (m/s)')
title('base velocity')
legend('true','estimated','Location','southwest')
grid on
ylim([-1 2])
subplot(2,2,2)
plot(time,trueVel(:,2),time,xest(:,5))
xlabel('time (s)')
ylabel('y velocity (m/s)')
title('base velocity')
legend('true','estimated','Location','southwest')
grid on
ylim([-1 2])
subplot(2,2,3)
plot(time,trueVel(:,3),time,xest(:,6))
xlabel('time (s)')
ylabel('z velocity (m/s)')
title('base velocity')
legend('true','estimated','Location','southwest')
grid on

figure
subplot(2,2,1)
plot(time,truePos(:,1),time,xest(:,1))
% for j = 1:length(contactInfo)-1
%     patchV = [contactInfo(j,1) -2; contactInfo(j,1) 2; contactInfo(j+1,1) 2; contactInfo(j+1,1) -2];
%     f = [1 2 3 4];
%     if contactInfo(j,2) == 1
%         color = 'red';
%     else
%         color = 'green';
%     end
%     patch('Faces',f,'Vertices',patchV,'FaceColor',color,'EdgeColor',color,'FaceAlpha',0.5)
% end
xlabel('time (s)')
ylabel('x distance (meters)')
title('base position')
legend('true','estimated','Location','northwest')
grid on
ylim([-0.5 2.5])
subplot(2,2,2)
plot(time,truePos(:,2),time,xest(:,2))
xlabel('time (s)')
ylabel('y distance (meters)')
title('base position')
legend('true','estimated','Location','northwest')
grid on
ylim([-0.5 2.5])
subplot(2,2,3)
plot(time,truePos(:,3),time,xest(:,3))
xlabel('time (s)')
ylabel('z distance (meters)')
title('base position')
legend('true','estimated','Location','southwest')
grid on

figure
subplot(2,2,1)
plot(time,world_leftFoot(:,1),time,xest(:,11))
xlabel('time (s)')
ylabel('x distance (meters)')
title('foot position')
legend('true','estimated','Location','northwest')
grid on
ylim([-0.5 2.5])
subplot(2,2,2)
plot(time,world_leftFoot(:,2),time,xest(:,12))
xlabel('time (s)')
ylabel('y distance (meters)')
title('foot position')
legend('true','estimated','Location','northwest')
grid on
ylim([-0.5 2.5])
subplot(2,2,3)
plot(time,world_leftFoot(:,3),time,xest(:,13))
xlabel('time (s)')
ylabel('z distance (meters)')
title('foot position')
legend('true','estimated','Location','southwest')
grid on
subplot(2,2,4)
plot(time,world_leftFoot(:,1),time,xest(:,11))
hold on
plot(time,world_leftFoot(:,2),time,xest(:,12))
plot(time,world_leftFoot(:,3),time,xest(:,13))

figure
plot(time,world_rightFoot(:,1),time,xest(:,14))
hold on
plot(time,world_rightFoot(:,2),time,xest(:,15))
plot(time,world_rightFoot(:,3),time,xest(:,16))





figure 
subplot(2,2,1)
plot(time,eul_true(:,1),time,eul_est(:,1))
xlabel('time (s)')
ylabel('z')
title('base orientation')
legend('true', 'estimated ','Location','southwest')
subplot(2,2,2)
plot(time,eul_true(:,2),time,eul_est(:,2))
xlabel('time (s)')
ylabel('y')
ylim([-0.5 0.5])
title('base orientation')
legend('true ', 'estimated ','Location','southwest')
subplot(2,2,3)
plot(time,eul_true(:,3),time,eul_est(:,3))
xlabel('time (s)')
ylabel('x')
ylim([-1 1])
title('base orientation')
legend('true ', 'estimated ','Location','southwest')
% subplot(2,2,4)
% plot(time,baseOrient(:,4),time,xest(:,10))
% xlabel('time (s)')
% ylabel('z')
% ylim([-0.5 0.5])
% title('base orientation')
% legend('true ', 'estimated ','Location','southwest')
%% test


%% function
function Ck = qRotationM(x)
        q0 = x(7,1);
        q1 = x(8,1);
        q2 = x(9,1);
        q3 = x(10,1);
%         
%         Ck = quat2rotm(x(7:10,1)');
        Ck = [2*(q0^2+q1^2)-1, 2*(q1*q2-q0*q3), 2*(q1*q3+q0*q2); 
            2*(q1*q2+q0*q3), 2*(q0^2+q2^2)-1, 2*(q2*q3-q0*q1);
            2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1), 2*(q0^2+q3^2)-1];
%         q = x(7:10,1);
%         Ck = [1-2*(q(3)*q(3)+q(4)*q(4)),2*(q(2)*q(3)-q(4)*q(1)),2*(q(2)*q(4)+q(3)*q(1));
%         2*(q(2)*q(3)+q(4)*q(1)),1-2*(q(2)*q(2)+q(4)*q(4)),2*(q(3)*q(4)-q(2)*q(1));
%         2*(q(2)*q(4)-q(3)*q(1)),2*(q(3)*q(4)+q(2)*q(1)),1-2*(q(2)*q(2)+q(3)*q(3))];
end

function qMap = rot2quat(deltaRot)
    qMap = [cos(1/2*norm(deltaRot));
            sin(1/2*(norm(deltaRot)))*deltaRot/(norm(deltaRot)) ];
end

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

function result = quaternionMultiplication(q1,q2)
    inter = q1(2)*q2(2)+q1(3)*q2(3)+q1(4)*q2(4);
    inter = inter(1);
    inter2_1_1 = [q1(1)*q2(2);q1(1)*q2(3);q1(1)*q2(4)];
    inter2_1_2 = [q2(1)*q1(2);q2(1)*q1(3);q2(1)*q1(4)];
    inter2_1 = inter2_1_1+inter2_1_2;
    inter2_2 = cross([q1(2);q1(3);q1(4)],[q2(2);q2(3);q2(4)]);
    result_1 = q1(1)*q2(1)-inter;
    result_2 = inter2_1 + inter2_2;
    result = [result_1;result_2];
end


% qMap01 = rot2quat(deltaRot)';
% q0 = qMap01(1);
% q1 = qMap01(2);
% q2 = qMap01(3);
% q3 = qMap01(4);

% qMap02 = rot2quat(angVel(3190,:)')';
% q0d = qMap02(1);
% q1d = qMap02(2);
% q2d = qMap02(3);
% q3d = qMap02(4);
% q = [q0*q0d-q1*q1d-q2*q2d-q3*q3d;
%     q0*q1d+q1*q0d+q2*q3d-q3*q2d;
%     q0*q2d-q1*q3d+q2*q0d+q3*q1d;
%     q0*q3d+q1*q2d-q2*q1d+q3*q0d;];


% fState = @(x)[x(1:3,1)+dt*x(4:6,1)+1/2*(dt^2*(linAcc'+g));  %p
%         x(4:6,1)+dt*(linAcc'+g);                            %v
%         (quatmultiply(rot2quat(angVel'*dt)',x(7:10,1)'))';  %q
%         x(11:13,1);                                         %p_leftFoot
%         x(14:16,1)];                                        %p_rightFoot
% xUpate = feval(f,xState);
% ykFun = @(xM)[leftFoot(:,1)-(xM(11:13,1)-xM(1:3,1));
%             rightFoot(:,1)-(xM(14:16,1)-xM(1:3,1))];


