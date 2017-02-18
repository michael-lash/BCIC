function [x obj]=sens_kernel(var)
[n,p]=size(var.X);
x=var.xinit;
l=var.l;
u=var.u;
% l=zeros(p,1);
% u=Inf(p,1);
diff=Inf;
Budget = var.B;


%indirectx=obj_kernel(x,var.indirectX,var.directX,var.indirectAlpha,var.indirectSigma);
%originalx = var.d'.*x+var.xbar';
%tempmatrix0 = kron(ones(size(var.directX,1),1),originalx')-var.directX;
%tempmatrix3 = exp(-kron(sum((tempmatrix0).^2,2),1/2./(var.indirectSigma.^2))).*var.indirectAlpha;
%tempmatrix1 = sum(var.indirectX.*tempmatrix3,1)';
%tempmatrix2 = sum(tempmatrix3);
%indirectx = tempmatrix1'./tempmatrix2;
%tempmatrix4=kron(ones(n,1),x')-var.X;
%tempmatrix5=kron(ones(n,1),indirectx)-var.indirectSV;
%temp=exp(-(sum((tempmatrix4).^2,2)+sum((tempmatrix5).^2,2))/2/var.sigma^2);
%temp=temp.*var.alpha.*var.y;
%obj=sum(temp);
originalx = var.d'.*x+var.xbar';
tempmatrix0 = kron(ones(size(var.directX,1),1),originalx')-var.directX;
tempmatrix3 = exp(-kron(sum((tempmatrix0).^2,2),1/2./(var.indirectSigma.^2))).*var.indirectAlpha;
tempmatrix1 = sum(var.indirectX.*tempmatrix3,1)';
tempmatrix2 = sum(tempmatrix3);
indirectx = tempmatrix1'./tempmatrix2;
obj=var.betachangeable'*originalx+var.betaindirect'*indirectx';


iter=1;
bestObj = obj;
bestObj
bestX = x;
while (Budget > 0 & iter < 100)
    

    for i=1:length(x)
        tempx = x;
        %Compute amount needed to satisfy B-x(i)*c(i) = 0
        purt = Budget/var.c(i);
        %Ensure that it's feasible
        if u(i) - (purt + tempx(i))  < 0
            purt = (u(i)-tempx(i));
        end
        if purt == 0
            continue
        end

        %Apply the changes
        tempx(i) = tempx(i) + purt;
        %Evaluate the changes

        originalx = var.d'.*tempx+var.xbar';
        tempmatrix0 = kron(ones(size(var.directX,1),1),originalx')-var.directX;
        tempmatrix3 = exp(-kron(sum((tempmatrix0).^2,2),1/2./(var.indirectSigma.^2))).*var.indirectAlpha;
        tempmatrix1 = sum(var.indirectX.*tempmatrix3,1)';
        tempmatrix2 = sum(tempmatrix3);
        indirectx = tempmatrix1'./tempmatrix2;
        tempobj=var.betachangeable'*originalx+var.betaindirect'*indirectx';


        %originalx = var.d'.*tempx+var.xbar';
        %tempmatrix0 = kron(ones(size(var.directX,1),1),originalx')-var.directX;
        %tempmatrix3 = exp(-kron(sum((tempmatrix0).^2,2),1/2./(var.indirectSigma.^2))).*var.indirectAlpha;
        %tempmatrix1 = sum(var.indirectX.*tempmatrix3,1)';
        %tempmatrix2 = sum(tempmatrix3);
        %indirectx = tempmatrix1'./tempmatrix2;
        %tempmatrix4=kron(ones(n,1),x')-var.X;
        %tempmatrix5=kron(ones(n,1),indirectx)-var.indirectSV;
        %temp=exp(-(sum((tempmatrix4).^2,2)+sum((tempmatrix5).^2,2))/2/var.sigma^2);
        %temp=temp.*var.alpha.*var.y;
        %tempobj=sum(temp);
        if tempobj < bestObj
            bestObj = tempobj;
            bestObj
            bestX = tempx;
        end
    end
    %Update x to be the bestX
    x= bestX;
    obj = bestObj;
    %Update the budget
    Budget = Budget - sum(var.c .* x);
    
    iter = iter + 1;

end

%while diff>var.tol && iter<=var.max_iter
    %iter
%    gradF=-var.indirectX.*tempmatrix3*diag(1./(var.indirectSigma).^2./tempmatrix2);
%    gradF=gradF+tempmatrix3*diag(1./(var.indirectSigma).^2./tempmatrix2)*diag(indirectx);
%    gradF=tempmatrix0'*gradF;
%    grad=sum(-diag(sparse(temp/var.sigma^2))*(tempmatrix4+tempmatrix5*gradF'*diag(var.d)),1)';
%    xnew=proj_truncsimplex(x-1/var.L*grad,var.c,var.B,l,u,var.gradtol);
%    originalx = var.d'.*x+var.xbar';
%    tempmatrix0 = kron(ones(size(var.directX,1),1),originalx')-var.directX;
%    tempmatrix3 = exp(-kron(sum((tempmatrix0).^2,2),1/2./(var.indirectSigma.^2))).*var.indirectAlpha;
%    tempmatrix1 = diag(var.indirectX'*tempmatrix3);
%    tempmatrix2 = sum(tempmatrix3);
%    indirectx = tempmatrix1'./tempmatrix2;
%    tempmatrix4=kron(ones(n,1),xnew')-var.X;
%    tempmatrix5=kron(ones(n,1),indirectx)-var.indirectSV;
%    temp=exp(-(sum((tempmatrix4).^2,2)+sum((tempmatrix5).^2,2))/2/var.sigma^2);
%    temp=temp.*var.alpha.*var.y;
%    while sum(temp)>obj(length(obj)) && var.L<1000
%        var.L=2*var.L;
%        xnew=proj_truncsimplex(x-1/var.L*grad,var.c,var.B,l,u,var.gradtol);
%        originalx = var.d'.*x+var.xbar';
%        tempmatrix0 = kron(ones(size(var.directX,1),1),originalx')-var.directX;
%        tempmatrix3 = exp(-kron(sum((tempmatrix0).^2,2),1/2./(var.indirectSigma.^2))).*var.indirectAlpha;
%        tempmatrix1 = diag(var.indirectX'*tempmatrix3);
%        tempmatrix2 = sum(tempmatrix3);
%        indirectx = tempmatrix1'./tempmatrix2;
%        tempmatrix4=kron(ones(n,1),xnew')-var.X;
%        tempmatrix5=kron(ones(n,1),indirectx)-var.indirectSV;
%        temp=exp(-(sum((tempmatrix4).^2,2)+sum((tempmatrix5).^2,2))/2/var.sigma^2);
%        temp=temp.*var.alpha.*var.y;
%    end
%    var.L=var.L/1.5;
%    x=xnew;
%    obj=[obj;sum(temp)];
%    diff=abs(sum(temp)-obj(length(obj)-1))/abs(obj(length(obj)-1));
%    iter=iter+1;
%end
