% Main function: knn_similarity_graph_full
function knn_similarity_graph_full(file_path, k_values, sigma, filename)

    % Load data based on file extension
    if endsWith(file_path, '.mat')
        data = load(file_path);
        field_names = fieldnames(data);
        X = data.(field_names{1});
    elseif endsWith(file_path, '.csv')
        X = readmatrix(file_path);
    else
        error('unsupported file format');
    end
    
    % Remove any cluster column in the data if present
    if size(X, 2) > 2
        X = X(:, 1:2);
    end
    
    % Calculate the Similarity Matrix S
    distances = pdist2(X, X, 'euclidean'); % Euclidean distance matrix
    S = exp(-distances.^2 / (2 * sigma^2)); % Similarity Matrix
    
    % Iterate over k_values to compute graphs and clusters
    for k = k_values
        W = compute_similarity_matrix(S, k); %Compute adjency matrix 
        D = create_degree_matrix(W); %Compute Degree matrix
        L = create_laplacian_matrix(D, W); % Compute Laplacian matrix
        
         % Compute eigenvalues and eigenvectors
        [U, D] = compute_eigenvalues(L);
        % Extract eigenvalues as a vector
        eigenvalues = diag(D);
        
        %plot eigenvalues
        figure;
        plot(eigenvalues, Marker="o");
        grid("on");
        title(sprintf('k= %d',k));
        
        % Determine number of clusters (M) from graph analysis
        if k == 10 && strcmp(filename, 'circle')
            threshold = 10^-2;
        elseif k == 20 && strcmp(filename, 'circle')
            threshold = 10^-1;
        elseif k == 40 && strcmp(filename, 'circle')
            threshold = 10^-1;
        elseif k == 10 && strcmp(filename, 'spiral')
            threshold = 10^-3;
        elseif k == 20 && strcmp(filename, 'spiral')
            threshold = 3 * 10^-3;
        elseif k == 40 && strcmp(filename, 'spiral')
            threshold = 3 * 10^-3;
        end
        M = sum(eigenvalues < threshold);
        % Select the first M eigenvectors
        U = U(:, 1:M); 
        % Perform k-means clustering on the reduced eigenvector space
        idx = kmeans(U, M);

        % Assign colors to clusters and visualize
        col = [0, 0.4470, 0.7410;
                  0.4660, 0.6740, 0.1880;
                  0.9290, 0.6940, 0.1250];
        colors = col(idx, :);
        figure;
        scatter(X(:,1), X(:,2), 15,colors, 'filled')

    end

    % compute and plot the kmeans function directly on X 
    idx_kmean = kmeans(X, M);
    figure;
    scatter(X(:,1), X(:,2), 15,colors(idx_kmean, :), 'filled')

    %compute and plot the single linkage function directly on X
    Z = linkage(X);
    c = cluster(Z,'Maxclust',3);
    figure;
    scatter(X(:,1),X(:,2),10,colors(c,:),'filled')
end

%Compute k-nearest neighbor similarity matrix
function W = compute_similarity_matrix(S, k)

    % Initialize adjacency matrix
    W = zeros(size(S));
    n = size(S, 1);
        
    % For each point, keep similarities to its k nearest neighbors
    for i = 1 : n
        [~, sortedIndices] = sort(S(i, :), 'descend'); % Sort neighbors by similarity
        sortedIndices = sortedIndices(1 : k); % Retain top k neighbors
        
        %Compute the symmetric matrix with only the top k neighbors
        for j = 1 : length(sortedIndices)
            W(i, sortedIndices(j)) = S(i, sortedIndices(j));
            W(sortedIndices(j), i) = S(i, sortedIndices(j));
        end
    end
    W = sparse(W); %Converse to spare matrix
end

%  Compute degree matrix
function D = create_degree_matrix(W)
    n = size(W, 1);
    D = zeros(n); % Initialize degree matrix

    % Compute the degree of each node
    for i = 1:n
        d_i = sum(W(i, :)); % Sum of weights for node i
        D(i, i) = d_i; % Assign degree to diagonal
    end
    D = sparse(D); %Converse to spare matrix
end

%Compute graph Laplacian Matrix
function L = create_laplacian_matrix(D, W)
    L = D - W;
end

%Compute eigenvalues and eigenvectors
function [U,D] = compute_eigenvalues(L)
% Compute the smallest 10 eigenvalues/eigenvectors based on Inverse Power Method
    [U, D] = eigs(L, 10, "smallestabs");
end



% Example usage
circle_path = 'Circle.csv';  % Path to circle dataset
spiral_path = 'Spiral.csv'; % Path to circle dataset
k_values = [10, 20, 40];  % Different k values for testing
sigma = 1;  % Gaussian kernel parameter

knn_similarity_graph_full(circle_path, k_values, sigma, 'circle');
knn_similarity_graph_full(spiral_path, k_values, sigma, 'spiral');


