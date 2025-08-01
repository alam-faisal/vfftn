function H = construct_Hamiltonian(hmodel,Q,par,varargin)

    opts = containers.Map;
    %defaults
    opts('mpo') = 'n';

    if nargin >= 4
        for n = 1:2:length(varargin)
            opts(varargin{n}) = varargin{n+1};
        end
    end


    if lower(opts('mpo')) == 'y'
        X = [0 1;1 0];Y = [0 -1.i;1.i 0];Z = [1 0;0 -1];
        switch lower(hmodel)
            case 'ising_1d_obc'
            J = par(1);
            g = par(2);

            H = cell(Q,1);

            for n = 2:Q-1
                m = zeros(3,3,2,2);
                m(1,1,:,:) = eye(2);
                m(2,1,:,:) = J*Z;
                m(3,1,:,:) = g*X;
                m(3,2,:,:) = Z;
                m(3,3,:,:) = eye(2);
                H{n} = m;
            end

            m = zeros(1,3,2,2);
            m(1,1,:,:) = g*X;
            m(1,2,:,:) = Z;
            m(1,3,:,:) = eye(2);
            H{1} = m;

            m = zeros(3,1,2,2);
            m(1,1,:,:) = eye(2);
            m(2,1,:,:) = J*Z;
            m(3,1,:,:) = g*X;
            H{Q} = m;

            case 'heisenberg_1d_obc'
            Jx = par(1);
            Jy = par(2);
            Jz = par(3);

            H = cell(Q,1);
            for n= 2:Q-1
                m = zeros(5,5,2,2);
                m(1,1,:,:) = eye(2);
                m(2,1,:,:) = Jx*X;
                m(3,1,:,:) = Jy*Y;
                m(4,1,:,:) = Jz*Z;
                m(5,2,:,:) = X;
                m(5,3,:,:) = Y;
                m(5,4,:,:) = Z;
                m(5,5,:,:) = eye(2);
                H{n} = m;
            end

            m = zeros(1,5,2,2);
            m(1,2,:,:) = X;
            m(1,3,:,:) = Y;
            m(1,4,:,:) = Z;
            m(1,5,:,:) = eye(2);
            H{1} = m;

            m = zeros(5,1,2,2);
            m(1,1,:,:) = eye(2);
            m(2,1,:,:) = Jx*X;
            m(3,1,:,:) = Jy*Y;
            m(4,1,:,:) = Jz*Z;
            H{Q} = m;

        end

    else


        switch lower(hmodel)

            case 'generic'
            pos = opts('pos');
            N = size(pos,1);
            H = zeros(N,Q+1);
            for n = 1:N
                H(n,pos(n,:)) = opts('type');
            end
            H(:,Q+1) = ones(N,1)*par(1);

            case 'field'
            J = par(1);
            H = opts('type')*eye(Q);
            H = [H J*ones(Q,1)];

            case '2d_nnn_pbc'
            Nx = par(1);
            Ny = par(2);
            J = par(3);

            tp = opts('type');

            H = [];
            %horizontal links
            for y = 1:Ny
                for x = 1:Nx-2
                    p1 = (y-1)*Nx + x;
                    p2 = p1 + 2;
                    h = zeros(1,Q+1);h(p1) = tp;h(p2) = tp;h(end) = J;
                    H = [H;h];
                end
                %boundary
                p1 = (y-1)*Nx + 1;
                p2 = y*Nx - 1;
                h = zeros(1,Q+1);h(p1) = tp;h(p2) = tp;h(end) = J;
                H = [H;h];
                p1 = (y-1)*Nx + 2;
                p2 = y*Nx;
                h = zeros(1,Q+1);h(p1) = tp;h(p2) = tp;h(end) = J;
                H = [H;h];
            end

            %vertical links
            for x = 1:Nx
                for y = 1:(Ny-2)
                    p1 = (y-1)*Nx + x;
                    p2 = (y+1)*Nx + x;
                    h = zeros(1,Q+1);h(p1) = tp;h(p2) = tp;h(end) = J;
                    H = [H;h];
                end
                %boundary
                p1 = x;
                p2 = (Ny-2)*Nx + x;
                h = zeros(1,Q+1);h(p1) = tp;h(p2) = tp;h(end) = J;
                H = [H;h];
                p1 = Nx + x;
                p2 = (Ny-1)*Nx + x;
                h = zeros(1,Q+1);h(p1) = tp;h(p2) = tp;h(end) = J;
                H = [H;h];

            end


            case '2d_nn_pbc'
            Nx = par(1);
            Ny = par(2);
            J = par(3);

            tp = opts('type');

            H = [];
            %horizontal links
            for y = 1:Ny
                for x = 1:Nx-1
                    p1 = (y-1)*Nx + x;
                    p2 = p1 + 1;
                    h = zeros(1,Q+1);h(p1) = tp;h(p2) = tp;h(end) = J;
                    H = [H;h];
                end
                %boundary
                p1 = (y-1)*Nx + 1;
                p2 = y*Nx;
                h = zeros(1,Q+1);h(p1) = tp;h(p2) = tp;h(end) = J;
                H = [H;h];
            end

            %vertical links
            for x = 1:Nx
                for y = 1:(Ny-1)
                    p1 = (y-1)*Nx + x;
                    p2 = y*Nx + x;
                    h = zeros(1,Q+1);h(p1) = tp;h(p2) = tp;h(end) = J;
                    H = [H;h];
                end
                %boundary
                p1 = x;
                p2 = (Ny-1)*Nx + x;
                h = zeros(1,Q+1);h(p1) = tp;h(p2) = tp;h(end) = J;
                H = [H;h];
            end



            case 'ising_1d_obc'
            J = par(1);
            g = par(2);
            % J*ZZ terms
            H = 3*eye(Q-1);
            H = H + diag(3*ones(Q-2,1),1);
            r = zeros(Q-1,1);r(end) = 3;
            H = [H r J*ones(Q-1,1)];
            % g*X terms
            H = [H; [eye(Q) g*ones(Q,1)]];

            case 'ising_1d_pbc'
            J = par(1);
            g = par(2);
            % J*ZZ terms
            H = 3*eye(Q);
            H = H + diag(3*ones(Q-1,1),1);
            H(Q,1) = 3;
            H = [H J*ones(Q,1)];
            % g*X terms
            H = [H; [eye(Q) g*ones(Q,1)]];

            case 'ising_1d_nnn_pbc'
            J1 = par(1);
            J2 = par(2);
            g = par(3);

            % J1*ZZ terms
            H = 3*eye(Q);
            H = H + diag(3*ones(Q-1,1),1);
            H(Q,1) = 3;
            H = [H J1*ones(Q,1)];

            % J2*ZZ terms
            h = 3*eye(Q);
            h = h + diag(3*ones(Q-2,1),2);
            h(Q-1,1) = 3;
            h(Q,2) = 3;

            H = [H; [h J2*ones(Q,1)]];

            % g*X terms
            H = [H; [eye(Q) g*ones(Q,1)]];

            case 'heisenberg_1d_obc'
            Jx = par(1);
            Jy = par(2);
            Jz = par(3);

            % Jx*XX terms
            H = 1*eye(Q-1);
            H = H + diag(1*ones(Q-2,1),1);
            r = zeros(Q-1,1);r(end) = 1;
            H = [H r Jx*ones(Q-1,1)];

            % Jy*YY terms
            h = 2*eye(Q-1);
            h = h + diag(2*ones(Q-2,1),1);
            r = zeros(Q-1,1);r(end) = 2;
            h = [h r Jy*ones(Q-1,1)];
            H = [H;h];

            % Jz*ZZ terms
            h = 3*eye(Q-1);
            h = h + diag(3*ones(Q-2,1),1);
            r = zeros(Q-1,1);r(end) = 3;
            h = [h r Jz*ones(Q-1,1)];
            H = [H;h];

            case 'heisenberg_1d_pbc'
            Jx = par(1);
            Jy = par(2);
            Jz = par(3);

            % Jx*XX terms
            H = 1*eye(Q);
            H = H + diag(1*ones(Q-1,1),1);
            H(Q,1) = 1;
            H = [H Jx*ones(Q,1)];

            % Jy*YY terms
            h = 2*eye(Q);
            h = h + diag(2*ones(Q-1,1),1);
            h(Q,1) = 2;
            h = [h Jy*ones(Q,1)];
            H = [H;h];

            % Jz*ZZ terms
            h = 3*eye(Q);
            h = h + diag(3*ones(Q-1,1),1);
            h(Q,1) = 3;
            h = [h Jz*ones(Q,1)];
            H = [H;h];

            otherwise
            disp('Model is not implemented')

        end

    end

end
