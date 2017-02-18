function [x obj]=GDNonCX_lgr_kernel(var)
[n,p]=size(var.X);
x=var.xinit;
l=var.l;
u=var.u;
% l=zeros(p,1);
% u=Inf(p,1);
diff=Inf;


%indirectx=obj_kernel(x,var.indirectX,var.directX,var.indirectAlpha,var.indirectSigma);
originalx = var.d'.*x+var.xbar';
tempmatrix0 = kron(ones(size(var.directX,1),1),originalx')-var.directX;
tempmatrix3 = exp(-kron(sum((tempmatrix0).^2,2),1/2./(var.indirectSigma.^2))).*var.indirectAlpha;
tempmatrix1 = sum(var.indirectX.*tempmatrix3,1)';
tempmatrix2 = sum(tempmatrix3);
indirectx = tempmatrix1'./tempmatrix2;
obj=var.betachangeable'*originalx+var.betaindirect'*indirectx';
iter=1;
while diff>var.tol && iter<=var.max_iter
    %iter
    gradF=-var.indirectX.*tempmatrix3*diag(1./(var.indirectSigma).^2./tempmatrix2);
    gradF=gradF+tempmatrix3*diag(1./(var.indirectSigma).^2./tempmatrix2)*diag(indirectx);
    gradF=tempmatrix0'*gradF;
    grad=var.d'.*var.betachangeable;
    grad=grad+(var.betaindirect'*gradF'*diag(var.d))';
    xnew=proj_truncsimplex(x-1/var.L*grad,var.c,var.B,l,u,var.gradtol);
    originalx = var.d'.*x+var.xbar';
    tempmatrix0 = kron(ones(size(var.directX,1),1),originalx')-var.directX;
    tempmatrix3 = exp(-kron(sum((tempmatrix0).^2,2),1/2./(var.indirectSigma.^2))).*var.indirectAlpha;
    tempmatrix1 = diag(var.indirectX'*tempmatrix3);
    tempmatrix2 = sum(tempmatrix3);
    indirectx = tempmatrix1'./tempmatrix2;
    temp=var.betachangeable'*originalx+var.betaindirect'*indirectx';
    while sum(temp)>obj(length(obj)) && var.L<1000
        var.L=2*var.L;
        xnew=proj_truncsimplex(x-1/var.L*grad,var.c,var.B,l,u,var.gradtol);
        originalx = var.d'.*xnew+var.xbar';
        tempmatrix0 = kron(ones(size(var.directX,1),1),originalx')-var.directX;
        tempmatrix3 = exp(-kron(sum((tempmatrix0).^2,2),1/2./(var.indirectSigma.^2))).*var.indirectAlpha;
        tempmatrix1 = diag(var.indirectX'*tempmatrix3);
        tempmatrix2 = sum(tempmatrix3);
        indirectx = tempmatrix1'./tempmatrix2;
        temp=var.betachangeable'*originalx+var.betaindirect'*indirectx';
    end
    var.L=var.L/1.5;
    x=xnew;
    obj=[obj;sum(temp)];
    diff=abs(sum(temp)-obj(length(obj)-1))/abs(obj(length(obj)-1));
    iter=iter+1;
end