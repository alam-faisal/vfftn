function H = ham2mat(F)

    op = cell(3,1);
    op{1} = [0 1;1 0];op{2} = [0 -1.i;1.i 0];op{3} = [1 0;0 -1];
    for n = 1:3
        op{n} = sparse(op{n});
    end

    Q = size(F,2)-1;


    H = sparse(2^Q,2^Q);
    for n = 1:size(F,1)
        pos = find(F(n,1:end-1));
        a = cell(length(pos),1);
        for k = 1:length(pos)
            a{k} = op{F(n,pos(k))};
        end

        H = H + F(n,end)*Kron(a,pos,Q);
    end


end


function out = Kron(op,pos,Q)

    Id = eye(2);Id = sparse(Id);

    count = 0;
    out = 1;
    for j = 1:Q
        if sum(pos == j) == 1
            count = count + 1;
            out = kron(op{count},out);
        else
            out = kron(Id,out);
        end
    end


end
