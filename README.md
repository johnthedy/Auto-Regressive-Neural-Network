# Auto-Regressive-Neural-Network
% Description

% This MATLAB script explain how Proposed ARNN (Auto-Regressive Neural Network) 
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
