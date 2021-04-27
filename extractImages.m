function [] = extractImages(Name, MOS, videos_path, frames_path)  
    parfor ind=1:len(Name)
        vidpath = strcat(frames_path, filesep, Name{ind,1});
        
        if MOS(ind) <= 1.8; fold = 'VeryPoor';
        elseif MOS(ind) <= 2.6; fold = 'Poor';
        elseif MOS(ind) <= 3.4; fold = 'Mediocre';
        elseif MOS(ind) <= 4.2; fold = 'Good';
        else; fold = 'VeryGood'; 
        end
        
        impath = strcat(frames_path, Name{ind,1}, filesep, fold);
        
        if(~exist(vidpath,'dir'))
            mkdir(vidpath)
            mkdir(impath);
        elseif(~exist(impath,'dir'))
            mkdir(impath);
        end
        
        v = VideoReader(char(strcat(videos_path, filesep, Name{ind,1}, '.mp4')));
        f = 0;
        while hasFrame(v)
            f = f+1;
            frame = readFrame(v);
            img = imresize(frame, [338 338]);
            img = imcrop(img, [19.5 19.5 298 298]);
            framepath = strcat(impath, filesep, sprintf('%04d',f), '.jpeg');
            imwrite(img, framepath,'Quality',100);
        end
            
        if(mod(ind,10)==0)
            disp(ind); 
        end
    end
end