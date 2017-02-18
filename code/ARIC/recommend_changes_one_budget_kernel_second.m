%function [x obj]=recommand(var)
load('~/ARIC/experiment/normDSetsStructAll.mat');
load('~/ARIC/experiment/logistic_model_multiple.mat');

var.indirectSV=supportVectors(:,indirectlyIndex);
var.directX=dSet1(:,changeableIndex);
var.indirectX=dSet1(:,indirectlyIndex);
var.indirectSigma=indirectSigma;
var.sigma=sigma;
var.B_item=4;
var.y=2*supportVectorLabels-1;
var.tol=0.0001;
var.gradtol=0.0001;
var.L=10;
var.max_iter=50;
ChangeMat=zeros(size(dSet2,1),length(changeableIndex));
for useid=1:size(dSet2,1)
    var.d=increaseCost;
    var.c=costChange';
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
    var.indirectAlpha=indirectAlpha;
    var.alpha=alpha_new;
    
    temp=dSet2(useid,changeableIndex);
    l=min(0,temp);
    u=max(1,temp);
    %var.l=-Inf(size(X_new,2),1);
    var.l=zeros(size(X_new,2),1);
    %var.l(find(var.d>0))=l(find(var.d>0))-temp(find(var.d>0));
    %var.l(find(var.d<0))=temp(find(var.d<0))-u(find(var.d<0));
    %var.u=Inf(size(X_new,2),1);
    %var.u=ones(size(X_new,2),1);
    var.u(find(var.d>0))=u(find(var.d>0))-temp(find(var.d>0));
    var.u(find(var.d<0))=temp(find(var.d<0))-l(find(var.d<0));
    
    %%%this part is only for the special requirement for smoking
    if dSet2(useid,changeableIndex(directionDependsInd(1)))==0
        var.u(directionDependsInd(1))=0;
    end
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
    %Prob(useid,1)=1./(1+exp(-LogisticModel{kFoldInd(useid)}.LogisticPara(1)-obj*LogisticModel{kFoldInd(useid)}.LogisticPara(2)));
    %prob_list=zeros(length(var.B_list),1);
    %for i=1:length(var.B_list) 
    var.B=var.B_item;
    [x obj]=GDNonCX_kernel(var);
    ChangeMat(useid,:)= x';%1./(1+exp(-LogisticModel{kFoldInd(useid)}.LogisticPara(1)-min(obj)*LogisticModel{kFoldInd(useid)}.LogisticPara(2)));
    [useid i]
    %end
end
%save('recommend_output_linear.mat','Prob');
save('~/ARIC/experiment/data_output/rec_chg_kernel_b4_second.mat','ChangeMat');
