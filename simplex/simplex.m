% Simplex algo
clear all
clc
close all
%%
data=csvread('LPtest.csv');
% data=csvread('test_data.csv');
%%
Aorig = data(2:end,2:end);
zorig = data(1,2:end);
borig = data(2:end,1);
totalCost = data(1,1);
clear data

[m,n]=size(Aorig);
%% Actual answer to compare against
options = optimset('LargeScale','off','Simplex','on');
[X,FVAL,EXITFLAG,OUTPUT]= linprog(zorig,[],[],Aorig,borig,zeros(size(zorig)),[],[],options);
%% Intro an atrificial basis with huge cost
BFScols = n+1:n+m;
Aaug = [Aorig eye(m) borig];
A = Aaug(:,1:end-1);
b = Aaug(:,end);
z = [zorig 99*ones(1,m) totalCost];
czorig = [zorig 99*ones(1,m)];
for i=1:m
    z = z - z(BFScols(i))*Aaug(i,:);
end
%% Iterate until all values in z are positive
loopCntr = 1; maxLoops = 200;
while(loopCntr < maxLoops)
%     display(['Loop: ' num2str(loopCntr)])
    
    % Find entering column by argmin(z)
    zminsIdx = 1;
    zmins = find(z==min(z(1:end-1)));
    enteringColIdx = zmins(zminsIdx);
    
    % Find exiting column by Theta rule.  If the theta has no finite
    % positive values greater than 0 then pick a new minimum cost column to
    % enter with and recalculate theta
    theta = b./Aaug(:,enteringColIdx);
    theta(Aaug(:,enteringColIdx)<=0)=nan;
    while (min(theta)==inf && zminsIdx<length(zmins))
        zminsIdx = zminsIdx+1;
        enteringColIdx = zmins(zminsIdx);
        
        theta = b./Aaug(:,enteringColIdx);
        theta(Aaug(:,enteringColIdx)<=0)=nan;
    end

    thetamins = find(theta==min(theta));
    % Select thetamin such that we pull the smallest i such that x_i is in
    % the BFS and check if Aaug(exitingBFSIdx, enteringColIdx) is 0
    for i=1:length(thetamins)
        exitingBFSIdx=find(BFScols==BFScols(thetamins(i)));
        if (Aaug(exitingBFSIdx, enteringColIdx) ~= 0)
            break;
        end
    end
    
    exitingColIdx = BFScols(exitingBFSIdx);
    Aaug(exitingBFSIdx,:) = Aaug(exitingBFSIdx,:)/Aaug(exitingBFSIdx, enteringColIdx);
    
    % We have normalized the correct row by the entering pivot, now clear
    % the column and adjust the costs
    for i=1:m
        if (i==exitingBFSIdx)
            continue
        end
        Aaug(i,:) = Aaug(i,:) - Aaug(i,enteringColIdx)*Aaug(exitingBFSIdx,:);
    end
    % Update A and b real quick
    A = Aaug(:,1:end-1);
    b = Aaug(:,end);
    z = z - z(enteringColIdx)*Aaug(exitingBFSIdx,:);
    BFScols(exitingBFSIdx) = enteringColIdx;
        
    if (all(z(1:end-1)>=0))
        break
    end
    loopCntr = loopCntr + 1;
end


x = zeros(n,1);
for i=1:m
    x(BFScols(i))=b(i);
end
g=sprintf('%.3f ', z);
display(['relative costs:   [' g ']'])
display(['bfs columns:   [' num2str(BFScols),']'])
g=sprintf('%.3f ', x);
display(['mine:   [' g,']'])
g=sprintf('%.3f ', X);
display(['theirs: [' g,']'])
display(['Finished looping after: ' num2str(loopCntr), ' loops'])