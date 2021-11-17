function [ data_train, data_query ] = getData_mine( MODE )
    % Generate training and testing data
    
    % Data Options:
    %   1. Toy_Gaussian
    %   2. Toy_Spiral
    %   3. Toy_Circle
    %   4. Caltech 101
    
    showImg = 1; % Show training & testing images and their image feature vector (histogram representation)
    
    PHOW_Sizes = [4 8 10]; % Multi-resolution, these values determine the scale of each layer.
    PHOW_Step = 8; % The lower the denser. Select from {2,4,8,16}
    
    switch MODE
            
        case 'Caltech' % Caltech dataset
            close all;
            %tic
            imgSel = [15 15]; % randomly select 15 images each class without replacement. (For both training & testing)
            folderName = './Caltech_101/101_ObjectCategories';
            classList = dir(folderName);
            classList = {classList(3:end).name} % 10 classes
            
            disp('Loading training images...')
            % Load Images -> Description (Dense SIFT)
            cnt = 1;
            if showImg
                figure('Units','normalized','Position',[.05 .1 .4 .9]);
                suptitle('Training image samples');
            end
            for c = 1:length(classList)
                subFolderName = fullfile(folderName,classList{c});
                imgList = dir(fullfile(subFolderName,'*.jpg'));
                imgIdx{c} = randperm(length(imgList));
                imgIdx_tr = imgIdx{c}(1:imgSel(1));
                imgIdx_te = imgIdx{c}(imgSel(1)+1:sum(imgSel));
                
                for i = 1:length(imgIdx_tr)
                    I = imread(fullfile(subFolderName,imgList(imgIdx_tr(i)).name));
                    
                    % Visualise
                    if i < 6 & showImg
                        subaxis(length(classList),5,cnt,'SpacingVert',0,'MR',0);
                        imshow(I);
                        cnt = cnt+1;
                        drawnow;
                    end
                    
                    if size(I,3) == 3
                        I = rgb2gray(I); % PHOW work on gray scale image
                    end
                    
                    % For details of image description, see http://www.vlfeat.org/matlab/vl_phow.html
                    [~, desc_tr{c,i}] = vl_phow(single(I),'Sizes',PHOW_Sizes,'Step',PHOW_Step); %  extracts PHOW features (multi-scaled Dense SIFT)
                end
            end
            
            disp('Building visual codebook...')
            % Build visual vocabulary (codebook) for 'Bag-of-Words method'
            desc_sel = single(vl_colsubset(cat(2,desc_tr{:}), 10e4)); % Randomly select 100k SIFT descriptors for clustering
            
            % K-means clustering
            numBins = 50; % for instance,
            
            % write your own codes here
            % ...
    
            % get center of clusters
            [~, centers] =  kmeans(desc_sel', numBins); 
                      
            disp('Encoding Images...')
            % Vector Quantisation
            
            % write your own codes here
            % ...
            % Bag of words dictrionary for training

            [R,C]=size(desc_tr);

            BOW_tr = zeros(R,C,numBins);
            
            for c = 1:length(classList)
                for i = 1:imgSel(1)
                    desc_size = size(desc_tr{c,i});
                    %BOW_tr{c,i} = zeros(1,numBins);
                    for j = 1:size(desc_size, 2)
                        [~, min_ind] = min(vl_alldist(single(desc_tr{c,i}(:,j)), single(centers'))) 
                        BOW_tr(c,i,min_ind) = BOW_tr(c,i,min_ind) + 1;
                    end
                end
            end

            data_train = BOW_tr;

            % calculate the sum of data_train
            %ans_tr = zeros(1, numBins);
            %for i = 1:10
              %  for j = 1:15
               %     ans_tr = ans_tr + data_train{i,j};
             %   end
            %end
            
            % Clear unused varibles to save memory
            clearvars desc_tr desc_sel
        %toc
    end
    
    switch MODE
        case 'Caltech'
            if showImg
            figure('Units','normalized','Position',[.05 .1 .4 .9]);
            suptitle('Test image samples');
            end
            disp('Processing testing images...');
            cnt = 1;
            % Load Images -> Description (Dense SIFT)
            %tic
            for c = 1:length(classList)
                subFolderName = fullfile(folderName,classList{c});
                imgList = dir(fullfile(subFolderName,'*.jpg'));
                imgIdx_te = imgIdx{c}(imgSel(1)+1:sum(imgSel));
                
                for i = 1:length(imgIdx_te)
                    I = imread(fullfile(subFolderName,imgList(imgIdx_te(i)).name));
                    
                    % Visualise
                    if i < 6 & showImg
                        subaxis(length(classList),5,cnt,'SpacingVert',0,'MR',0);
                        imshow(I);
                        cnt = cnt+1;
                        drawnow;
                    end
                    
                    if size(I,3) == 3
                        I = rgb2gray(I);
                    end
                    [~, desc_te{c,i}] = vl_phow(single(I),'Sizes',PHOW_Sizes,'Step',PHOW_Step);
                
                end
            end
            %suptitle('Testing image samples');

            %if showImg
            %    figure('Units','normalized','Position',[.5 .1 .4 .9]);
            %    suptitle('Testing image representations: 256-D histograms');
            %end
    
            % Quantisation
            
            % write your own codes here
            % ...

            BOW_te = zeros(R,C,numBins);
            
            for c = 1:length(classList)
                for i = 1:imgSel(1)
                    desc_size = size(desc_te{c,i});
                    %BOW_te{c,i} = zeros(1,numBins);
                    for j = 1:size(desc_size, 2)
                        [~, min_ind] = min(vl_alldist(single(desc_te{c,i}(:,j)), single(centers'))) 
                        BOW_te(c,i,min_ind) = BOW_te(c,i,min_ind) + 1;
                    end
                end
            end   
            
            data_query = BOW_te;
        %toc
        % calculate accuracy
        % 100*numel(find(data_train==data_query))/numel(data_train)       
    end
    
    