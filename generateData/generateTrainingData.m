function Data = generateTrainingData(M,Data,data_num,num_on_triPF, num_on_invtriPF, num_on_random,seed, mode)
% 这个函数是在PF上采样来构造solution set，
% -----------task-------------------
% dataset_num 100,000 -> 1,000,000
% dataset:
%   正三角PF上: 50,000
%   倒三角PF上: 50,000
%   random generated: 900,000
% -----------task-------------------
    a = 1.5;
    b = 0.5; % a和b两个参数是控制下面的p的取值范围
    rng(seed);    
    tic
    % generate solutions for triangular and inverted triangular PF. 
    for i=1:num_on_triPF
        if mod(i, 10000) == 0
            disp(['i=',num2str(i),'/',num2str(num_on_triPF+num_on_invtriPF+num_on_random)]);
            toc
        end
        num = 0;
        while num <= 0
            num = ceil(data_num*rand);
        end
        
        p = rand*a+b; % p是用来控制PF的曲率
        temp = abs(UniformSphere_ExponentioalPowerDistribution(num,ones(1,M)*p,1));
        Data(i,1:num,:) = temp';
    end
    for i=num_on_triPF+1:num_on_triPF+num_on_invtriPF
        if mod(i, 10000) == 0
            disp(['i=',num2str(i),'/',num2str(num_on_triPF+num_on_invtriPF+num_on_random)]);
            toc
        end
        num = 0;
        while num <= 0
            num = ceil(data_num*rand);
        end
        p = rand*a+b; % p是用来控制PF的曲率
        temp = abs(UniformSphere_ExponentioalPowerDistribution(num,ones(1,M)*p,1));
        temp = temp*(-1)+1;
        Data(i,1:num,:) = temp';
        %HVC(:,i) = CalHVC(Data(:,:,i),r,data_num);  
    end

    % generate random non-dominated solutions. 
    %% 先生成一个很大很大的solutions集合
    archiveSize = 100*data_num*M;
    solutions = rand(archiveSize, M);
    % get non-dominated solutions
    [FrontNo,MaxFNo] = NDSort(solutions, archiveSize);
    disp(['generate a large solution set and perform NDSort, seed=', num2str(seed)])
    disp(['MaxFNo=', num2str(MaxFNo)])
    toc
    
    %% 按照mode来选取selected_ndsolutions with num
    for i=num_on_triPF+num_on_invtriPF+1:num_on_triPF+num_on_invtriPF+num_on_random
%         if mod(i, 10000) == 0
%             disp(['i=',num2str(i),'/',num2str(num_on_triPF+num_on_invtriPF+num_on_random)]);
%             toc
%         end
        
        % the dataset contain num of points
        num = 0;
        while num <= 0       % <=0 for [1, 100], <=100 for [101, 200]
            num = ceil(data_num*rand);
        end
%         disp(['num=', num2str(num)])
        
        % 按照mode选取一个front并且确保选取的front里的solution数大于等于num。
        % 得到ndsolutions
        check_num_points = 0;
        while check_num_points < num
            if strcmp(mode, 'random')
                selectedNo = ceil(MaxFNo*rand); 
            elseif strcmp(mode, 'worst')
                probability = 1:MaxFNo;
                probability = cumsum(probability / sum(probability));
                selectedNo = sum(rand >= probability);
            elseif strcmp(mode, 'best')
                probability = MaxFNo:-1:1;
                probability = cumsum(probability / sum(probability));
                selectedNo = sum(rand >= probability);
            end
            ndsolutions = solutions(FrontNo==selectedNo,:); 
            check_num_points = size(ndsolutions, 1);
        end

        % randomly select num of points. 
        selected_ndsolutions = ndsolutions(randperm(size(ndsolutions,1),num),:);
        Data(i,1:num,:) = selected_ndsolutions;
    end
end