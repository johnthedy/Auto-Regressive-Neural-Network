%% Description
% This script explain how Proposed ARNN (Auto-Regressive Neural Network) 
% works. This script model a 6 story MDOF using state space method where
% first segment (Start) consist of input variable for mass, damping, and
% stiffness for each story (base to level 5).

% Second segment (Structure characteristic) is construction of state space
% matrix

%Third segment (Linear time history analysis using state space formulation)
%shows an artificial ground motion generation using white gaussian noise
%with 50s duration and 0.005s increment for state space model ground motion 
%input. Then it continue with state space simulation.

%Fourth segment (Training) shows how previous ground motion and response
%output data is organized into training data for Neural Network.

%Last segment (Validation) validate the accuracy of trained ARNN. An
%earthquakes ground motion data from newcEQ1.mat is used as testing data to
%validate the performance of trained ARNN.

%% Start
clc;clear;
nstory=6;
%Mass each floor
mb=6000;m1=6000;m2=6000;m3=6000;m4=6000;m5=6000;
%Stiffness each floor
kb=4000000;k1=4000000;k2=4900000;k3=4900000;k4=4900000;k5=4900000;
%Damping each floor
cb=40000;c1=60000;c2=50000;c3=50000;c4=50000;c5=40000;

%Property of structure
M=[mb,0,0,0,0,0;0,m1,0,0,0,0;0,0,m2,0,0,0;0,0,0,m3,0,0;0,0,0,0,m4,0;0,0,0,0,0,m5];
K=[k1+kb,-k1,0,0,0,0;-k1,k1+k2,-k2,0,0,0;0,-k2,k2+k3,-k3,0,0;0,0,-k3,k3+k4,-k4,0;0,0,0,-k4,k4+k5,-k5;0,0,0,0,-k5,k5];
C=[c1+cb,-c1,0,0,0,0;-c1,c1+c2,-c2,0,0,0;0,-c2,c2+c3,-c3,0,0;0,0,-c3,c3+c4,-c4,0;0,0,0,-c4,c4+c5,-c5;0,0,0,0,-c5,c5];

%% Structure characteristic
%SYSTEM MATRIX
SA=[zeros(nstory) eye(nstory);-M\K -M\C];
SB=vertcat(zeros(nstory,1),-1*ones(nstory,1));
SC=[-M\K -M\C];
SD=-1*ones(nstory,1);

%% Linear time history analysis using state space formulation
%Input Earthquake or Motion t=time (s), gm=ground motion (m/s2)
EQinc=0.005;
t=0:EQinc:50;
gm=wgn(length(t),1,0);gm(1)=0;
gm=gm./max(abs(gm));

sys=ss(SA,SB,SC,SD);
[~,~,x]=lsim(sys,gm,t);
x6=x(:,6).*1000;

%% Training
nw1=150;nw2=25;
D=vertcat(zeros(nw1,1),x6);
GM=vertcat(zeros(nw2-1,1),gm);

for i=1:length(D)-nw1-1
    X1(i,:)=D(i:i+nw1-1);
    Y(i,:)=D(i+nw1);
end
for i=1:length(GM)-nw2
    X2(i,:)=GM(i:i+nw2-1);
end

FX=horzcat(X1,X2);FY=Y;
Mdl=fitrnet(FX,FY,"Standardize",true,"LayerSizes",[10 10 10],'Activations','relu',"Verbose",1,'IterationLimit',2e3);

%% Verification
%Input Earthquake or Motion t=time (s), gm=ground motion (m/s2)
load('newcEQ1.mat')
t=newcEQ1(:,1);
gm=newcEQ1(:,2);

%Simulink Calculation
sys=ss(SA,SB,SC,SD);
[y,tout,x]=lsim(sys,gm,t);
x6=x(:,6).*1000;

%AI Prediction
GM=vertcat(zeros(nw2,1),gm);
X1=zeros(1,nw1);
for i=1:length(gm)
    X2=GM(i:i+nw2-1)';
    Ypred=predict(Mdl,horzcat(X1,X2));
    X1=horzcat(X1(2:end),Ypred);
    R(i)=Ypred;
end

%% Plot Result
hold on
plot(x6)
plot(R)
grid on
hold off
legend('Simulink Result','AI Prediction','Location','best')
x0=10;y0=10;width=500;height=250;
set(gcf,'position',[x0,y0,width,height])
xlabel('Time (s)', 'Fontsize',10);
ylabel('Displacement (mm)', 'Fontsize',10);
set(gca,'fontname','times')