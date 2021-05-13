%% Botond Nas (KHUPIB) -- Watershed Segmentation

%% import image
rgb = imread('sample.jpg');
if size(rgb,3)>2 % if image is rgb (3 color channel), convert to grayscale
    I = rgb2gray(rgb);
else
    I = rgb;
end
I = im2double(I); % convert from int to double
% I = double(1) - I;
O = reconstruction(I,11,0.19); % reconstruction function
%imshow(O)

%% Opening-Closing
% Opening
mat1 = erosion(O,11);
mat2 = dilation(mat1,11);

% Closing
mat3 = dilation(mat2, 11);
mat4 = erosion(mat3, 11);
%imshow(mat4)

%% Gradient Magnitude of the original grayscale image
gmag = gradient_magnitude(I);
% imshow(gmag)
% title('Gradient Magnitude')

%% Finding Local Minima of the image
isMin = islocalmin(mat4,1) & islocalmin(mat4,2);
%isMin = islocalmin(O,1) & islocalmin(O,2);
[row,col] = find(isMin); % indicies of all the minima
%imshow(isMin)
figure, montage({I,O,mat4,gmag})

%% Watershed process

A = createLabelledPixels(gmag);
for i = 1:size(row, 1)
    A(row(i, 1),col(i, 1),1) = i; % each minima assigned a unique CB
    A(row(i, 1),col(i, 1),3) = 1; % mark each minima as 'added to queue'
end

B=watershed(A, row, col);
segments = B(:,:,1);

% color each segment a different random color
final = zeros(size(segments,1), size(segments,2),3);
for i = 1:size(row,1)
    R = rand;
    G = rand;
    B = rand;
    for j = 1:size(segments,1)
        for k = 1:size(segments,2)
            if segments(j,k) == i
                final(j,k,:)=[R G B];
            elseif segments(j,k) == -1
                final(j,k,:)=[1 1 1];
            end
        end
    end
end
%imshow(final)
figure, montage({I, final})
title('Final Segmentation')

%% Functions
function A = createLabelledPixels(gmag)
    A = zeros(size(gmag,1), size(gmag,2),3);
    for i = 1:size(gmag, 1)
        for j = 1:size(gmag, 2)
            A(i,j,1) = 0; % label (which basin it's a member of)
            A(i,j,2) = gmag(i,j); % gradient magnitude
            A(i,j,3) = 0; % has been in queue marker
        end
    end
end

% input B matrix and coordinates: ex. C = [200, 100]
function output = findUnmarkedNeighbors(B, C)
    if C(1) == 1
        if C(2) == 1
            output = zeros(2,3);
            output(1,:) = [C(1), C(2)+1, B(C(1), C(2)+1,3)];
            output(2,:) = [C(1)+1, C(2), B(C(1)+1, C(2),3)];
        elseif C(2) == size(B, 2)
            output = zeros(2,3);
            output(1,:) = [C(1), C(2)-1, B(C(1), C(2)-1,3)];
            output(2,:) = [C(1)+1, C(2), B(C(1)+1, C(2),3)];
        else
            output = zeros(3,3);
            output(1,:) = [C(1), C(2)-1, B(C(1), C(2)-1,3)];
            output(2,:) = [C(1), C(2)+1, B(C(1), C(2)+1,3)];
            output(3,:) = [C(1)+1, C(2), B(C(1)+1, C(2),3)];
        end
    elseif C(1) == size(B,1)
        if C(2) == 1
            output = zeros(2,3);
            output(1,:) = [C(1), C(2)+1, B(C(1), C(2)+1,3)];
            output(2,:) = [C(1)-1, C(2), B(C(1)-1, C(2),3)];
        elseif C(2)==size(B,2)
            output = zeros(2,3);
            output(1,:) = [C(1), C(2)-1, B(C(1), C(2)-1,3)];
            output(2,:) = [C(1)-1, C(2), B(C(1)-1, C(2),3)];
        else
            output = zeros(3,3);
            output(1,:) = [C(1), C(2)-1, B(C(1), C(2)-1,3)];
            output(2,:) = [C(1), C(2)+1, B(C(1), C(2)+1,3)];
            output(3,:) = [C(1)-1, C(2), B(C(1)-1, C(2),3)];
        end
    elseif C(2) == 1
        output = zeros(3,3);
        output(1,:) = [C(1), C(2)+1, B(C(1), C(2)+1,3)];
        output(2,:) = [C(1)-1, C(2), B(C(1)-1, C(2),3)];
        output(3,:) = [C(1)+1, C(2), B(C(1)+1, C(2),3)];
    elseif C(2) == size(B,2)
        output = zeros(3,3);
        output(1,:) = [C(1), C(2)-1, B(C(1), C(2)-1,3)];
        output(2,:) = [C(1)-1, C(2), B(C(1)-1, C(2),3)];
        output(3,:) = [C(1)+1, C(2), B(C(1)+1, C(2),3)];
    else
        output = zeros(4,3);
        output(1,:) = [C(1), C(2)-1, B(C(1), C(2)-1,3)];
        output(2,:) = [C(1), C(2)+1, B(C(1), C(2)+1,3)];
        output(3,:) = [C(1)-1, C(2), B(C(1)-1, C(2),3)];
        output(4,:) = [C(1)+1, C(2), B(C(1)+1, C(2),3)];
    end
    
    output(output(:,3)==1, :)=[];
end

% returns -1 if labelled neighbors' labels are not equal, otherwise returns
% the CB number (uniform marker) of the neighbors
function label = areNeighborLabelsEqual(B, C)
    label = -1;
    if C(1) == 1
        if C(2) == 1
            result = zeros(2,3);
            result(1,:) = [C(1), C(2)+1, B(C(1), C(2)+1,1)];
            result(2,:) = [C(1)+1, C(2), B(C(1)+1, C(2),1)];
        elseif C(2) == size(B, 2)
            result = zeros(2,3);
            result(1,:) = [C(1), C(2)-1, B(C(1), C(2)-1,1)];
            result(2,:) = [C(1)+1, C(2), B(C(1)+1, C(2),1)];
        else
            result = zeros(3,3);
            result(1,:) = [C(1), C(2)-1, B(C(1), C(2)-1,1)];
            result(2,:) = [C(1), C(2)+1, B(C(1), C(2)+1,1)];
            result(3,:) = [C(1)+1, C(2), B(C(1)+1, C(2),1)];
        end
    elseif C(1) == size(B,1)
        if C(2) == 1
            result = zeros(2,3);
            result(1,:) = [C(1), C(2)+1, B(C(1), C(2)+1,1)];
            result(2,:) = [C(1)-1, C(2), B(C(1)-1, C(2),1)];
        elseif C(2)==size(B,2)
            result = zeros(2,3);
            result(1,:) = [C(1), C(2)-1, B(C(1), C(2)-1,1)];
            result(2,:) = [C(1)-1, C(2), B(C(1)-1, C(2),1)];
        else
            result = zeros(3,3);
            result(1,:) = [C(1), C(2)-1, B(C(1), C(2)-1,1)];
            result(2,:) = [C(1), C(2)+1, B(C(1), C(2)+1,1)];
            result(3,:) = [C(1)-1, C(2), B(C(1)-1, C(2),1)];
        end
    elseif C(2) == 1
        result = zeros(3,3);
        result(1,:) = [C(1), C(2)+1, B(C(1), C(2)+1,1)];
        result(2,:) = [C(1)-1, C(2), B(C(1)-1, C(2),1)];
        result(3,:) = [C(1)+1, C(2), B(C(1)+1, C(2),1)];
    elseif C(2) == size(B,2)
        result = zeros(3,3);
        result(1,:) = [C(1), C(2)-1, B(C(1), C(2)-1,1)];
        result(2,:) = [C(1)-1, C(2), B(C(1)-1, C(2),1)];
        result(3,:) = [C(1)+1, C(2), B(C(1)+1, C(2),1)];
    else
        result = zeros(4,3);
        result(1,:) = [C(1), C(2)-1, B(C(1), C(2)-1,1)];
        result(2,:) = [C(1), C(2)+1, B(C(1), C(2)+1,1)];
        result(3,:) = [C(1)-1, C(2), B(C(1)-1, C(2),1)];
        result(4,:) = [C(1)+1, C(2), B(C(1)+1, C(2),1)];
    end
    
    result(result(:,3)==0, :)=[];
    if ~isempty(result) && all(~diff(result(:,3))) % returns logical 1 (true) if all elements in the 3rd column are equal
        label = result(1,3);
    end
end

function A = watershed(A, row, col)
    % create a PriorityQueue where the 'priority' is the third element
    % (gradient magnitude)
    q = PriorityQueue(3);
    % insert all unmarked neighbors of the regional minima to the queue
    for i = 1:size(row,1)
        n = findUnmarkedNeighbors(A, [row(i,1), col(i,1)]);
        for k = 1:size(n, 1)
            A(n(k,1), n(k,2), 1) = A(row(i,1), col(i,1), 1);
            A(n(k,1), n(k,2), 3) = 1; % change 'added to queue' marker to 1
            q.insert([n(k,1), n(k,2), A(n(k,1), n(k,2), 2)]);        
        end
    end
    % while the queue is not empty
    while(q.size() > 0)
        popped = q.remove();
        A(popped(1,1), popped(1,2),1)=areNeighborLabelsEqual(A, [popped(1,1), popped(1,2)]);
        n = findUnmarkedNeighbors(A, [popped(1,1), popped(1,2)]);
        for k = 1:size(n, 1)
            A(n(k,1),n(k,2),3)=1; % change 'added to queue' marker to 1
            q.insert([n(k,1), n(k,2), A(n(k,1), n(k,2), 2)]); % insert into queue     
        end
        q.size()
    end
end

function gmag = gradient_magnitude(I)
    gmag = zeros(size(I, 1), size(I, 2));
    for r = 2:size(I, 1)-1    % for number of rows of the image
        for c = 2:size(I, 2)-1    % for number of columns of the image        
            gmag(r,c)=sqrt((((I(r+1,c)-I(r-1,c))/2)^2) + (((I(r,c+1)-I(r,c-1))/2)^2));
        end
    end
end

function mat = erosion(I, seSize)    
    % create structuring element              
    se=ones(seSize, seSize);

    % store number of rows in P and number of columns in Q.            
    [P, Q]=size(se); 

    % create a zero matrix of size I.        
    mat=zeros(size(I, 1), size(I, 2)); 

    for i=ceil(P/2):size(I, 1)-floor(P/2)
        for j=ceil(Q/2):size(I, 2)-floor(Q/2)

            % take all the neighbourhoods.
            on=I(i-floor(P/2):i+floor(P/2), j-floor(Q/2):j+floor(Q/2));
            mat(i,j)=min(on(:));
        end
    end
end

function mat = dilation(I, seSize)
    % create structuring element             
    se=ones(seSize, seSize); 

    % store number of rows in P and number of columns in Q.           
    [P, Q]=size(se); 

    % create a zero matrix of size I.        
    mat=zeros(size(I, 1), size(I, 2)); 

    for i=ceil(P/2):size(I, 1)-floor(P/2)
        for j=ceil(Q/2):size(I, 2)-floor(Q/2)

            % take all the neighbourhoods.
            on=I(i-floor(P/2):i+floor(P/2), j-floor(Q/2):j+floor(Q/2));  
            mat(i,j)=max(on(:));
        end
    end
end

% function bin = toBinary(I, thresh)
%     % create a zero matrix of size I.        
%     bin=zeros(size(I, 1), size(I, 2)); 
%     
%     for r = 1:size(I, 1)    % for number of rows of the image
%         for c = 1:size(I, 2)    % for number of columns of the image        
%             if I(r,c)<thresh
%                 bin(r,c)=0;
%             else
%                 bin(r,c)=1;
%             end
%         end
%     end
% end

function O = reconstruction(mask, seSize, prop)
    marker = zeros(size(mask,1),size(mask,2));
    O = zeros(size(mask,1),size(mask,2));
    O_old = ones(size(mask,1),size(mask,2));
    sub = max(mask, [], 'all')*prop;
    %sub = min(mask, [], 'all')*prop;
    
    for r = 1:size(marker, 1)    % for number of rows of the image
        for c = 1:size(marker, 2)    % for number of columns of the image        
            marker(r,c)=mask(r,c)-sub;
            %marker(r,c)=mask(r,c)+sub;
        end
    end
    
    while O_old ~= O
        % create structuring element             
        se=ones(seSize, seSize); 

        % store number of rows in P and number of columns in Q.           
        [P, Q]=size(se); 

        for i=ceil(P/2):size(marker, 1)-floor(P/2)
            for j=ceil(Q/2):size(marker, 2)-floor(Q/2)

                % take all the neighbourhoods.
                on=marker(i-floor(P/2):i+floor(P/2), j-floor(Q/2):j+floor(Q/2));
                maxV = min(on(:));
                if maxV >= mask(i,j)
                    %O(i,j)=maxV;
                    O(i,j)=mask(i,j);
                else
                    %O(i,j)=mask(i,j);
                    O(i,j)=maxV;
                end
                %marker(i,j)=max(on(:));
            end
        end
        O_old = O;
    end
end