% save movie as .mat from chuncks
path = '~/Documents/datasets/vid075-chunks';
movie = load_movie(path, 56, 128, 64);
save('movie.mat', 'movie')
fprintf('saved!');

function movie = load_movie(path, num_chunks, im_sz, len_chunk)
    movie = zeros(im_sz, im_sz, len_chunk, num_chunks);
    for i=1:num_chunks
      movie(:,:,:,i) = ...
        read_chunk(path, i, im_sz, len_chunk); 
    end
end

function chunk = read_chunk(path, i, im_sz, len_chunk)
    filename=sprintf('%s/chunk%d',path,i);
    fprintf('%s\n',filename);
    fid=fopen(filename,'r','b');
    chunk=reshape(fread(fid,im_sz*im_sz*len_chunk,'float'),im_sz,im_sz,len_chunk);
    fclose(fid);
end
