similarity={};
for i=1:10
    testInd=find(kFoldInd~=i);
    similarity{i}=zeros(size(testInd,1),1);
    for j=1:size(testInd,1)
        [i,j]
        temp=exp(-sum((kFoldSV(i)-kron(ones(size(kFoldSV(i),1),1),dSet2(testInd(j),:))).^2,2)/2/kFoldSigma(i)^2);
        similarity{i}(j)=sum(temp.*kFoldAlpha(i).*(2*kFoldLabel(i)-1));
        %similarity{i}(j)=sum(temp.*kFoldAlpha(i).*kFoldLabel(i));
    end
end
LogisticModel={}
for i=1:10
    i
    [LogisticModel{i}.LogisticPara,LogisticModel{i}.dev,LogisticModel{i}.stats] =mnrfit(similarity{i},-dSet2Label(find(kFoldInd~=i))+2)
end