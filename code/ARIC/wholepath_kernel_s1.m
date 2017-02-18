load('~/ARIC/experiment/normDSetsStructAll.mat');
load('~/ARIC/experiment/logistic_model_multiple.mat');

B_list=1:20;
useid=randi([1,size(dSet2,1)])
useid=15
var.d=increaseCost;
var.c=costChange';
for i=1:length(directionDependsInd)
    if dSet2(useid,changeableIndex(directionDependsInd(i)))<=directionDependsCutoff(i)
        var.d(directionDependsInd(i))=1;
    else
        var.d(directionDependsInd(i))=-1;
    end
end

[alpha_new,X_new,indirectAlpha]=preprocess_kernel(supportVectors,dSet1(:,unchangeableIndex),dSet2(useid,:),var.d,alphaVals,changeableIndex,unchangeableIndex,sigma,indirectSigma);
var.X=X_new;
var.xbar=dSet2(useid,changeableIndex);
var.indirectSV=supportVectors(:,indirectlyIndex);
var.directX=dSet1(:,changeableIndex);
var.indirectX=dSet1(:,indirectlyIndex);
var.indirectSigma=indirectSigma;
var.indirectAlpha=indirectAlpha;
[n,p]=size(var.X);
var.sigma=sigma;
var.alpha=alpha_new;
var.y=2*supportVectorLabels-1;
var.tol=0.00001;
var.gradtol=0.000001;
var.L=10;
%var.xinit=rand(size(var.X,2),1)*var.B./var.c/size(var.X,2);
var.xinit=zeros(size(var.X,2),1);
var.max_iter=50;
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

prob_list=zeros(length(B_list),1);
x_list=zeros(size(var.X,2),length(B_list));
x=zeros(p,1);
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
objPGD=sum(temp);
for i=1:length(B_list) 
    B_list(i)
    var.B=B_list(i);
    [x obj]=GDNonCX_kernel(var);
    objPGD=[objPGD;min(obj)];
    prob_list(i)=1./(1+exp(-LogisticModel{kFoldInd(useid)}.LogisticPara(1)-min(obj)*LogisticModel{kFoldInd(useid)}.LogisticPara(2)));
%     if (x(binaryChangeable)>0 && x(binaryChangeable)<1)
%         print 'binary variable is not 0 or 1'
%     end
    var.xinit=x;
    %var.xinit=rand(size(var.X,2),1)*var.B./var.c/size(var.X,2);
    x_list(:,i)=x;
end

direct = var.d;
save('~/ARIC/experiment/data_output/changes_p15_kernel_setting1.mat','x_list','direct');

% figure
% plot(B_list,prob_list)
% hold on
% figure
% plot(x_list')
figure
changeid=find(sum(abs(x_list),2)>0);
plot((diag(var.d(changeid))*x_list(changeid,:))')
hold on
legend(header(changeableIndex(changeid)))
