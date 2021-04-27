function [acts] = extractVideoFeatures(idsmap, videos, net)
    acts = struct;
    files = cell(len(videos),1);
    features = cell(len(videos),1);
    
    for i=1:len(videos)
        tmp = activations(net,idsmap(videos(i)),'avg_pool');
        files{i} = videos(i);
        features{i} = reshape(tmp,[size(tmp,3),size(tmp,4)]);
        
        if(mod(i,10)==0)
            disp(i); 
        end
    end
    
    acts.Files = [files{:}]';
    acts.Features_full = features;
    
    acts.Features_avg = cell2mat(cellfun(@(x) (reshape(mean(x,2),[1,len(x)])),acts.Features_full,'UniformOutput',false));
    acts.Features_median = cell2mat(cellfun(@(x) (reshape(median(x,2),[1,len(x)])),acts.Features_full,'UniformOutput',false));
    acts.Features_min = cell2mat(cellfun(@(x) (reshape(min(x,[],2),[1,len(x)])),acts.Features_full,'UniformOutput',false));
    acts.Features_max = cell2mat(cellfun(@(x) (reshape(max(x,[],2),[1,len(x)])),acts.Features_full,'UniformOutput',false));
end