%% Clear Workspace
% 
clear all
clc
%% Load and Transform Data
Y15 = importfile("gemini_BTCUSD_2015_1min.csv");
Y16 = importfile("gemini_BTCUSD_2016_1min.csv");
Y17 = importfile("gemini_BTCUSD_2017_1min.csv");
Y18 = importfile("gemini_BTCUSD_2018_1min.csv");
Y19 = importfile("gemini_BTCUSD_2019_1min.csv");
Y20 = importfile("gemini_BTCUSD_2020_1min.csv");
Y21 = importfile("gemini_BTCUSD_2021_1min.csv");

%Slim down Data to Time and Close
Yfull= [Y15(:,[2 7]);Y16(:,[2 7]);Y17(:,[2 7]);Y18(:,[2 7]);Y19(:,[2 7]);Y20(:,[2 7]);Y21(:,[2 7])];
%Yi is the main data used
Yi= [Y19(257798:end,[2 7]);Y20(:,[2 7]);Y21(:,[2 7])];
T=(length(Yi.Close));



%% Show Raw Bitcoin Data

figure (1)
plot(Yi.Date, Yi.Close)
ytickformat('usd')


%% Check for missing values

% This section finds the areas with missing values and stores the date and
% length of the missing sections
b=1;

for i=1:T-1
    
diffm = minutes(Yi.Date(i) - Yi.Date (i+1));
if diffm == -1

else
    DIFF(b,:) = [diffm string(Yi.Date(i))];
    b = b+1;
end

end    
%% RETIME fill missing values with linear equation

% First the data not used gets choped off (section with large gaps) and
% stored in the timeTabel X

X = table2timetable([Y19(257798:end,[2 7]);Y20(:,[2 7]);Y21(:,[2 7])]);

%Fill missing values with a linear Methode
Y = retime(X, 'minutely', 'linear');
Tt = length(Y.Close);

% Index missing values and create a mask of wich values are missing
missing = retime(X,'minutely','fillwithmissing');
mask = ~isnan(missing.Close);



%% Reach Stationarity

%Take the difference

%In a first step I take the natural log of the data
yl = log(Y.Close);

%In a next stepp I take the difference 
dy = diff(yl);

%% Creating the training data

%Defining the training data with the length of four weeks
ytrain = yl(1:40320); %28days
Ts = length(ytrain);

% Test with Dickey-Fuller-Test for stationarity
[h] = adftest(y, alpha = 0.01)
[h]= kpsstest(y, alpha = 0.01) 

% Plot ACF and PACF
figure(3)
subplot(3,1,1)
plot(dy)
subplot(3,1,2) 
autocorr(dy)
subplot(3,1,3)
parcorr(dy)


% With the one difference the time series is now stationary.

%% Model estimate (chose D)

%This section tests to find the best d according to the AIC and BIC values
%and then stores the value for the best d. The estimated models
%only use the training data.

MaxD = 2;
p=0;
q=0;
AICBIC = [];
for d = 0:MaxD
    Mdl = arima(0,d,0);
      EM = estimate(Mdl,ytrain,'display','off').summarize;
 
       AICBIC(d+1,1) = EM.BIC;
       AICBIC(d+1,2) = EM.AIC;
       AICBIC(d+1,3) = d;
    
end    


[row,~]=find(min(AICBIC(:,2)) == AICBIC);
BestDaic = row;
[row,~]=find(min(AICBIC(:,1)) == AICBIC);
BestDbic = row;

if BestDaic == BestDbic
    BestD = BestDaic-1;
else
    BestD = BestDaic-1;
end

%% Choose parameter for P and Q via best BIC/AIC

%This test to the defined range MaxP and MaxQ to find the model with the
%lowest AIC/BIC. The value of d has to be definded before this step. The
%results get stored in BestP and BestQ respectively. The estimated models
%only use the training data.

%The parfor loop inside the outer loop allows for parallel computation to
%speed up the calculations

tic
BestD = 1
MaxP = 20;
MaxQ = 20;
BIC =[];
AIC =[];

for p = 1:MaxP+1
    parfor q = 1:MaxQ+1
        Mdl = arima(p-1,BestD,q-1);
        EM = estimate(Mdl,ytrain, 'display','off').summarize
        BIC(p,q) =EM.BIC;
        AIC(p,q)= EM.AIC;

    end
end

[AICp,AICq]=find(min(AIC,[],'all') == AIC);
[BICp,BICq]=find(min(AIC,[],'all') == AIC);

if AICp == BICp
    BestP = AICp-1;
else
    BestP = AICp-1;
    disp("BIC and AIC do not aggree, AIC chosen as default")
end

if AICq == BICq
    BestQ = AICq-1;
else
     BestQ = AICq-1;
     disp("BIC and AIC do not aggree, AIC chosen as default")
end

toc
%Source for why AIC prefered over BIC if they should disagree
%https://robjhyndman.com/hyndsight/to-explain-or-predict/






%% Creat Working Model
%Crating the Model with the training data
WkMdl = arima(BestP,BestD,BestQ);
WkEstMdl = estimate(WkMdl,ytrain);

% Godness of fit
%Testing the fit of the model

%Creating the residuals and standardised residuals based on the
%trainingdata
residuals = infer(WkEstMdl,ytrain);
stres = residuals/sqrt(WkEstMdl.Variance);

% Ljung-Box Q-test for residual autocorrelation
[h pValue]  = lbqtest(residuals)

%Ploting the residuals for visual inseption
figure (5)
subplot(2,2,1)
plot(stres)
title('Standardized Residuals')
subplot(2,2,2)
qqplot(residuals)
subplot(2,2,3)
autocorr(residuals)
subplot(2,2,4)
parcorr(residuals)

%% Forecasting

%This section forecasts the value over the defind range. The values get
%stored in the Matrix A together with a timestamp

tic

% Set how long the forecast should be:
dd = 1;

% how far in the past the loop should go:
jj = Tt-Ts-1;

%shorter forcast for testing
% jj = 40000;

%The parfor loop works the same way as a for loop but can do the
%calculations on multiple cores in parallel

A = zeros(2,jj);
parfor i = 1:jj
A(:,i) = [forecast(WkEstMdl,dd,yl(i:Ts+i-1)) Ts+i];  
end
A= flip(A)';

toc

%% Check direction of prediction

%This section tests if the forcasted value moves in the same direction ans
%the observed value (up or down) and stores them in the vetor results.


results=[];
for i = 1:jj;

%     future value minus past value
 if yl(Ts+i)-yl(Ts-1+i) > 0; yreal = 1; else yreal = -1; end % one means goes up, negative means goes down
 
 if A(i,2) - yl(Ts-1+i) >0; ypredict=1; else ypredict = -1; end
 
 % fills the list backwards, today towards the past
 if yreal == ypredict; results(i,1)=1; else results(i,1)=0; end
 
end

figure (8)
histogram(results)
%% Correcting results for the filled in values

%This section uses the created mask vektor to drop the results from the
%sections I initialy filled in and stores the new results in the vector
%cleanresults. 

cleanedresults = results(mask(Ts+1:jj+Ts));
inverscleandresults = results(~mask(Ts+1:jj+Ts));

figure (9)
histogram(cleanedresults)


%% Statistical Tests

%How many percent correct
PercentCorrect = sum(results)/length(results);
PercentCorrect = sum(cleanedresults)/length(cleanedresults);

%Runstest
[h, p] = runstest(results)
[h, p] = runstest(cleanedresults)


%Binominal Distribution
l= length(results);
lc= length(cleanedresults);
x = sum(results);
xc = sum(cleanedresults);

Chanceofmore = 1-binocdf(x,l,0.5)
Chanceofmorecleand = 1-binocdf(xc,lc,0.5)

binocdf(xc,lc,0.5,'upper')
