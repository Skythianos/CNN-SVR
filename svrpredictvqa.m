function [PLCC, SROCC, KROCC, YPRED, Y_test] = svrpredictvqa(TrainVideos, ValidationVideos, TestVideos, Name, MOS, actsstruct)
    
    t = struct2table(actsstruct,'RowNames',actsstruct.Files);
    t2 = table(MOS, 'RowNames', Name);

    for i=1:4
    
        if i==1
            X_train = t([TrainVideos; ValidationVideos],:).Features_avg;
            X_test = t(TestVideos,:).Features_avg;
        elseif i==2
            X_train = t([TrainVideos; ValidationVideos],:).Features_median;
            X_test = t(TestVideos,:).Features_median;
        elseif i==3
            X_train = t([TrainVideos; ValidationVideos],:).Features_min;
            X_test = t(TestVideos,:).Features_min;
        elseif i==4
            X_train = t([TrainVideos; ValidationVideos],:).Features_max;
            X_test = t(TestVideos,:).Features_max;
        end
        Y_train = t2([TrainVideos; ValidationVideos],:).MOS;
        Y_test = t2(TestVideos,:).MOS;

        TrainVideoLevelFeatures = X_train;
        TrainMOS = Y_train;

        TestVideoLevelFeatures = X_test;
        TestMOS = Y_test;

        MdlGan = fitrsvm(TrainVideoLevelFeatures, TrainMOS, 'Standardize', true, 'KernelFunction', 'gaussian', 'KernelScale', 'auto');

        YPredGan = predict(MdlGan, TestVideoLevelFeatures);

        P = corr(YPredGan, TestMOS, 'Type', 'Pearson');
        S = corr(YPredGan, TestMOS, 'Type', 'Spearman');
        K = corr(YPredGan, TestMOS, 'Type', 'Kendall');
        YP = YPredGan;
        
        if i==1
            PLCC.avg = P;
            SROCC.avg = S;
            KROCC.avg = K;
            YPRED.avg = YP;
        elseif i==2
            PLCC.median = P;
            SROCC.median = S;
            KROCC.median = K;
            YPRED.median = YP;
        elseif i==3
            PLCC.min = P;
            SROCC.min = S;
            KROCC.min = K;
            YPRED.min = YP;
        elseif i==4
            PLCC.max = P;
            SROCC.max = S;
            KROCC.max = K;
            YPRED.max = YP;
        end
    
    end
    
end