function [alpha_new,X_new,indirectAlpha]=preprocess_kernel(X,unchangedX,xbar,d,alpha,changeableIndx,unchangeableIndx,sigma,indirectSigma)
%unchangeableIndx=setdiff(1:size(X,2),changeableIndx);
indirectAlpha=unchangedX-kron(ones(size(unchangedX,1),1),xbar(unchangeableIndx));
indirectAlpha=exp(-kron(sum(indirectAlpha.*indirectAlpha,2),1/2./(indirectSigma.^2)));
X_new=(X(:,changeableIndx)-kron(ones(size(X,1),1),xbar(changeableIndx)))*diag(d);
alpha_new=alpha.*exp(-sum((X(:,unchangeableIndx)-kron(ones(size(X,1),1),xbar(unchangeableIndx))).^2,2)/2/sigma^2);
