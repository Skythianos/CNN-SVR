function [imageDatastoreObj] = filterFrames(idsmap, videos, frac)
    f = {};
    l = {};

    for i=1:len(videos)
        m = idsmap(videos(i)).copy;
        rp = randperm(len(m.Files),round(len(m.Files)*frac));
        f = [f; m.Files(rp)];
        l = [l; m.Labels(rp)];
    end

    imageDatastoreObj = m.copy;
    l = cellstr(l);
    [~, I] = sort(l);
    imageDatastoreObj.Files = f(I);
    imageDatastoreObj.Labels = categorical(l(I));
end