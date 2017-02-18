function lgr_glmnet(dataFile,saveFile)

load(dataFile)

options1.alpha=0;
l2glmnetcv = cvglmnet(dSet1,dSet1Label,'binomial',options1);
%cvglmnetPlot(l2glmnetcv);
options1.lambda=l2glmnetcv.lambda_min;
l2glmnet = glmnet(dSet1,dSet1Label,'binomial',options1);

options2.alpha=1;
l1glmnetcv = cvglmnet(dSet1,dSet1Label,'binomial',options2);
%cvglmnetPlot(tempglmnet);
options2.lambda=l1glmnetcv.lambda_min;
l1glmnet = glmnet(dSet1,dSet1Label,'binomial',options2);

options3.alpha=1;
options3.lambda=0;
finalglmnet = glmnet(dSet1(:,find(l1glmnet.beta)),dSet1Label,'binomial',options3);

save(saveFile);
