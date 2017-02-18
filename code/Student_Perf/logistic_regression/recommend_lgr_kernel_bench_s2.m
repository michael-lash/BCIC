%function [x obj]=recommand(var)
load('~/ARIC/experiment/benchmark/logistic_regression/glmnet_model_bench.mat')
var.indirectSV=supportVectors(:,indirectlyIndex);
var.directX=dSet1(:,changeableIndex);
var.indirectX=dSet1(:,indirectlyIndex);
var.indirectSigma=indirectSigma;
var.sigma=sigma;
var.B_list=1:20;
var.y=2*supportVectorLabels-1;
var.tol=0.0001;
var.gradtol=0.0001;
var.L=10;
var.max_iter=50;
var.d=increaseCost;
var.c=costChange';
betasparse=l1glmnet.beta;
betasparse(find(l1glmnet.beta))=finalglmnet.beta;
var.betachangeable=betasparse(changeableIndex);
var.betaindirect=betasparse(indirectlyIndex);
Prob=zeros(size(dSet2,1),length(var.B_list)+1);
for useid=1:size(dSet2,1)
    for i=1:length(directionDependsInd)
        if dSet2(useid,changeableIndex(directionDependsInd(i)))<=directionDependsCutoff(i)
            var.d(directionDependsInd(i))=1;
        else
            var.d(directionDependsInd(i))=-1;
        end
    end
    %%centerize data and decision varaible
    [alpha_new,X_new,indirectAlpha]=preprocess_kernel(supportVectors,dSet1(:,unchangeableIndex),dSet2(useid,:),var.d,alphaVals,changeableIndex,unchangeableIndex,sigma,indirectSigma);
    var.X=X_new;
    var.xbar=dSet2(useid,changeableIndex);
    var.xunchange=dSet2(useid,unchangeableIndex);
    var.indirectAlpha=indirectAlpha;
    var.alpha=alpha_new;
    
    temp=dSet2(useid,changeableIndex);
    l=min(0,temp);
    u=max(1,temp);
    %var.l=-Inf(size(X_new,2),1);
    var.l=zeros(size(X_new,2),1);
    var.l(find(var.d>0))=l(find(var.d>0))-temp(find(var.d>0));
    var.l(find(var.d<0))=temp(find(var.d<0))-u(find(var.d<0));
    %var.u=Inf(size(X_new,2),1);
    %var.u=ones(size(X_new,2),1);
    var.u(find(var.d>0))=u(find(var.d>0))-temp(find(var.d>0));
    var.u(find(var.d<0))=temp(find(var.d<0))-l(find(var.d<0));
    
    %%%this part is only for the special requirement for smoking
    %if dSet2(useid,changeableIndex(directionDependsInd(1)))==0
    %    var.u(directionDependsInd(1))=0;
    %end
    %%%

    var.xinit=zeros(size(var.X,2),1);
    x=var.xinit;
    [n,p]=size(var.X);
    originalx = var.d'.*x+var.xbar';
%     tempmatrix0 = kron(ones(size(var.directX,1),1),originalx')-var.directX;
%     tempmatrix3 = exp(-kron(sum((tempmatrix0).^2,2),1/2./(var.indirectSigma.^2))).*var.indirectAlpha;
%     tempmatrix1 = sum(var.indirectX.*tempmatrix3,1)';
%     tempmatrix2 = sum(tempmatrix3);
%     indirectx = tempmatrix1'./tempmatrix2;
    indirectx = dSet2(useid,indirectlyIndex);
    tempmatrix4=kron(ones(n,1),x')-var.X;
    tempmatrix5=kron(ones(n,1),indirectx)-var.indirectSV;
    temp=exp(-(sum((tempmatrix4).^2,2)+sum((tempmatrix5).^2,2))/2/var.sigma^2);
    temp=temp.*var.alpha.*var.y;
    obj=sum(temp);
    Prob(useid,1)=1/(1+exp(-dSet2(useid,find(betasparse))*finalglmnet.beta-finalglmnet.a0));
    %prob_list=zeros(length(var.B_list),1);
    for i=1:length(var.B_list) 
        var.B=var.B_list(i);
        [x obj]=GDNonCX_lgr_kernel(var);
        originalx = var.d'.*x+var.xbar';
        tempmatrix0 = kron(ones(size(var.directX,1),1),originalx')-var.directX;
        tempmatrix3 = exp(-kron(sum((tempmatrix0).^2,2),1/2./(var.indirectSigma.^2))).*var.indirectAlpha;
        tempmatrix1 = diag(var.indirectX'*tempmatrix3);
        tempmatrix2 = sum(tempmatrix3);
        indirectx = tempmatrix1'./tempmatrix2;
        temp=zeros(size(dSet2,2),1);
        temp(changeableIndex)=originalx;
        temp(indirectlyIndex)=indirectx;
        temp(unchangeableIndex)= var.xunchange;
        Prob(useid,i+1)=1/(1+exp(-temp'*betasparse-finalglmnet.a0));
        [useid i]
    end
end
%save('~/ARIC/experiment/benchmark/logistic_regression/results/recommend_output_kernel_bench_s1.mat','Prob'); %w/ var.l commented out
save('~/ARIC/experiment/benchmark/logistic_regression/results/recommend_output_kernel_bench_s2.mat','Prob'); %w/ var.l uncommented out
