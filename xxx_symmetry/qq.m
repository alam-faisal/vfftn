clear all

Q = 10;
U = basis_change(Q);


lattice_data = containers.Map;
lattice_data('Q') = Q;
lattice_data('NN') = [1:Q-1;2:Q]';
lattice_data('triangles') = [];%[1 2 3];
H = build_H(lattice_data,0.3);
H = ham2mat(H);
Hp = H;
H = U'*H*U;

H(abs(H) < 1.e-10) = 0;

% H = sparse(H);
B = divide_into_blocks(H,Q);
Sm = size(B,1);
o = cell(1,Sm);
for l = 1:Sm
    z = divide_by_id(B{l,l},2*l-1);
    o{l} = eig(z);
end

for n = 1:Sm
    writematrix(sort(o{n}),['data/Q=' num2str(Q) '_S=' num2str(n-1) '.txt'])
end


function B = divide_into_blocks(M,n)

    d = degW(n);
    N = length(d);
    B = cell(N,N);

    if mod(n,2) == 1
        w = 2*[1:N];
    else
        w = [0:N-1]*2+1;
    end

    d = d.*w;

    for k = 1:N
        for l = 1:N
            d1b = 1+sum(d(1:k-1));
            d1e = sum(d(1:k));
            d2b = 1+sum(d(1:l-1));
            d2e = sum(d(1:l));
            B{k,l} = M(d1b:d1e,d2b:d2e);
        end
    end



end

function g = swap

    g = zeros(2,2,2,2);
    g(1,1,1,1) = 1;
    g(1,2,2,1) = 1;
    g(2,1,1,2) = 1;
    g(2,2,2,2) = 1;
    g = reshape(g,4,4);
end

function Mp = apply_TM(M,n)
    % M is chi x chi matrix, only relevant blocks are stored (without identity)
    % M is multiplied by A_n, A^*_n transfer matrix

    if mod(n,2) == 1
        N = (n+1)/2;
        Mp = cell(N,1);
        for k = 1:(N-1)
            Mp{k} = blkdiag(M{k},M{k+1});
        end
        Mp{N} = M{N};
    else
        N = 1+n/2;
        Mp = cell(N,1);
        Mp{1} = M{1};
        for k = 2:(N-1)
            Mp{k} = blkdiag(M{k},M{k-1});
        end
        Mp{N} = M{N-1};
    end

end

function Hf = fullH(H,n)

    if mod(n,2) == 0
        N = 1+n/2;
        d = zeros(N,1);
        for k = 1:N
            d(k) = 2*k-1;
        end
    else
        N = (n+1)/2;
        d = zeros(N,1);
        for k = 1:N
            d(k) = 2*k;
        end
    end

    a = cell(N,1);
    for n = 1:N
        a{n} = kron(H{n},speye(d(n)));
    end

    Hf = blkdiag(a{:});

end


function U = basis_change(Q)
    A = cell(1,Q);
    for n = 1:Q
        A{n} = fulltensor(n);
    end

    lc = {};
    lc{1} = [-Q-2 1 -1];
    for n = 2:Q-1
        lc{n} = [n-1 n -n];
    end
    lc{Q} = [Q-1 -Q-1 -Q];

    U = scon(A,lc);
    U = reshape(U,2^Q,2^Q);

end




function G = divide_by_id(K,n)

    dimleft = size(K,1)/n;

    G = zeros(dimleft);
    for k = 1:dimleft
        for l = 1:dimleft
            G(k,l) = K(1 + (k-1)*n,1 + (l-1)*n);
        end
    end

end


function H = build_H(data,J_chi)

    Q = data('Q');
    NN = data('NN');
    triangles = data('triangles');

    H = [];
    %Heisenberg term
    for k = 1:size(NN,1)
        for n = 1:3
            h = construct_Hamiltonian('generic',Q,  1/4,...
            'pos',NN(k,:),'type',[n n]);
            H = [H;h];
        end
    end

    %traiangle terms
    for k = 1:size(triangles,1)
        h1 = construct_Hamiltonian('generic',Q,     J_chi/8,...
        'pos',triangles(k,:),'type',[1 2 3]);
        h2 = construct_Hamiltonian('generic',Q,     -J_chi/8,...
        'pos',triangles(k,:),'type',[1 3 2]);
        h3 = construct_Hamiltonian('generic',Q,     J_chi/8,...
        'pos',triangles(k,:),'type',[2 3 1]);
        h4 = construct_Hamiltonian('generic',Q,     -J_chi/8,...
        'pos',triangles(k,:),'type',[2 1 3]);
        h5 = construct_Hamiltonian('generic',Q,     J_chi/8,...
        'pos',triangles(k,:),'type',[3 1 2]);
        h6 = construct_Hamiltonian('generic',Q,     -J_chi/8,...
        'pos',triangles(k,:),'type',[3 2 1]);

        H = [H;h1;h2;h3;h4;h5;h6];
    end
end



function A = fulltensor(n)

    A = zeros(2^(n-1),2^n,2);

    dL = degW(n-1);
    dR = degW(n);

    if mod(n,2) == 0
        bldimsL = (2*[1:length(dL)]).*dL;
        bldimsR = (2*[1:length(dR)]-1).*dR;

        for l = 1:2
            m = -l+3/2;
            % diagonal elements
            for k = 1:length(bldimsL)
                pL1 = 1+sum(bldimsL(1:k-1));
                pL2 = pL1+bldimsL(k)-1;
                pR1 = 1+sum(bldimsR(1:k-1));
                pR2 = pR1+bldimsR(k)-1;
                A(pL1:pL2,pR1:pR2,l) = ...
                kron([eye(dL(k)) zeros(dL(k),dR(k)-dL(k))],...
                CGCfull(k-1/2,k-1,m));
            end
            % superdiagonal
            for k = 2:length(bldimsR)
                pL1 = 1+sum(bldimsL(1:k-2));
                pL2 = pL1+bldimsL(k-1)-1;
                pR1 = 1+sum(bldimsR(1:k-1));
                pR2 = pR1+bldimsR(k)-1;
                A(pL1:pL2,pR1:pR2,l) = ...
                kron([zeros(dL(k-1),dR(k)-dL(k-1)) eye(dL(k-1))],...
                CGCfull(k-3/2,k-1,m));
            end

        end

    else

        bldimsL = (2*[1:length(dL)]-1).*dL;
        bldimsR = (2*[1:length(dR)]).*dR;


        for l = 1:2 %physical dimension
            m = -l+3/2;
            % diagonal elements
            for k = 1:length(bldimsL)
                pL1 = 1+sum(bldimsL(1:k-1));
                pL2 = pL1+bldimsL(k)-1;
                pR1 = 1+sum(bldimsR(1:k-1));
                pR2 = pR1+bldimsR(k)-1;
                A(pL1:pL2,pR1:pR2,l) = ...
                kron([eye(dL(k)) zeros(dL(k),dR(k)-dL(k))],...
                CGCfull(k-1,k-1/2,m));
            end

            %subdiagonal
            for k = 2:length(bldimsL)
                pL1 = 1+sum(bldimsL(1:k-1));
                pL2 = pL1+bldimsL(k)-1;
                pR1 = 1+sum(bldimsR(1:k-2));
                pR2 = pR1+bldimsR(k-1)-1;
                A(pL1:pL2,pR1:pR2,l) = ...
                kron([zeros(dL(k),dR(k-1)-dL(k)) eye(dL(k))],...
                CGCfull(k-1,k-3/2,m));
            end
        end
    end
end


function C = CGCfull(j1,j2,m)

    c = cgc(j1,j2,m);
    C = zeros(2*j1+1,2*j2+1);

    if m > 0
        if j1 > j2
            for n = 1:length(c)
                C(n+1,n) = c(n);
            end
        else
            for n = 1:length(c)
                C(n,n) = c(n);
            end
        end
    else
        if j1 > j2
            for n = 1:length(c)
                C(n,n) = c(n);
            end
        else
            for n = 1:length(c)
                C(n,n+1) = c(n);
            end
        end
    end


end



function a = cgc(j1,j2,m)

    a = [];
    if m > 0
        if j1 > j2
            for m2 = j2:-1:(-j2)
                a(end+1) = -sqrt(0.5*(1 - m2/(j1+0.5) ));
            end
        else
            for m2 = j2:-1:(-j2+1)
                a(end+1) = sqrt(0.5*(1 + m2/(j1+0.5) ));
            end
        end
    else
        if j1 > j2
            for m2 = j2:-1:(-j2)
                a(end+1) = sqrt(0.5*(1 + m2/(j1+0.5) ));
            end
        else
            for m2 = j2-1:-1:(-j2)
                a(end+1) = sqrt(0.5*(1 - m2/(j1+0.5) ));
            end
        end
    end
end

function out = degW(n)

    a = [1];

    for k = 1:(n/2)
        b = [a 0] + [0 a];b = b(2:end);
        a = [b 0] + [0 b];
    end

    if mod(n,2) == 0
        out = a;
    else
        b = [a 0] + [0 a];b = b(2:end);
        out = b;
    end

end

% Define the range of Q values
Q_values = [2,3,4, 6, 8, 10];  % Example: modify for your desired Q values

% Loop over each Q value
for Q = Q_values
    % Compute the matrix U using the basis_change function
    U = basis_change(Q);

    % Save the resulting matrix to a text file (CSV or plain text)
    filename = sprintf('N=%d.txt', Q);
    writematrix(U, filename);  % Save matrix in plain text format
end
