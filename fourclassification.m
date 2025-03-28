% Use the stochastic parallel gradient descent algorithm to perform a four-class classification task
%% Obtain the handle of the corresponding device
FPGA=serial('com13'); %choose serial port
set(FPGA,'BaudRate',115200,'StopBits',1,'DataBits',8);
fopen(FPGA);
sprintf('FPGA COM port has been opened.');

%PD 
PD=serial('com14'); %choose serial port
set(PD,'BaudRate',115200,'StopBits',1,'DataBits',8);
fopen(PD);
%% Generate the training dataset
load MNIST.mat
V_ub_data = 4;  %The voltage value corresponding to a π phase shift in this experiment.
V_lb_data = 0;

label = [0 1 5 8];  %Define the dataset labels for the classification task.
batch_size_train = 50*length(label);  %Define the num for the classification task.
batch_num = 1;   %Define the number of training batches

%Randomly select a specific number of images.
data_x_train = [];
data_y_train = [];
for i = 1 : length(label)
    a = find(train_image_label == label(i))';
    data_x_train = [data_x_train, a( randperm(length(a), batch_num * batch_size_train / length(label)) )];
    data_y_train = [data_y_train, label(i) * ones(1, batch_num * batch_size_train / length(label) )];
end
data = [data_x_train; data_y_train];

data_x_train = zeros(64, 1,1);
data_y_train = zeros(1,1,1);

%Process the data to generate the training dataset
xuhao = randperm(batch_num * batch_size_train);
count = 1;
for i = 1:batch_num
    for j =  1:batch_size_train
        a = train_image_mnist(:,:,data(1, xuhao(count)));
        mid = imresize(a, [8, 8], 'bilinear');
        mid =  reshape(mid, [], 1);
        mid = mid/max(mid);
        data_x_train(:,j, i) = mid * (V_ub_data - V_lb_data) + V_lb_data;
        data_y_train(1, j, i)=data(2, xuhao(count));
        count = count + 1;       
    end
end
%% Generate the testing dataset
batch_size_test = 30*length(label);

data_x_test = [];
data_y_test = [];
for i = 1 : length(label)
    a = find(test_image_label == label(i))';
    data_x_test = [data_x_test, a( randperm(length(a), batch_num * batch_size_test / length(label)) )];
    data_y_test = [data_y_test, label(i) * ones(1, batch_num * batch_size_test / length(label) )];
end
data = [data_x_test; data_y_test];

data_x_test = zeros(64, 1,1);
data_y_test = zeros(1,1,1);

xuhao = randperm(batch_num * batch_size_test);
count = 1;
for i = 1:batch_num
    for j =  1:batch_size_test
        a = test_image_mnist(:,:,data(1, xuhao(count)));
        mid = imresize(a, [8, 8], 'bilinear');
        mid =  reshape(mid, [], 1);
        mid = mid/max(mid);
        data_x_test(:,j, i) = mid * (V_ub_data - V_lb_data) + V_lb_data;
        data_y_test(1, j, i)=data(2, xuhao(count));
        count = count + 1;
    end
end

%% Define the thermal phase shifter indices, with each index corresponding to a specific thermal phase shifter on the chip
CHmod1(1,:) = [113 114 115 116 117 118 119 120 1 2 3 4 5 6 7 8];
CHmod1(2,:) = [137 138 139 140 141 142 143 144 9 10 11 12 13 14 15 16];
CHmod1(3,:) = [129 130 131 132 133 134 135 136 17 18 19 20 21 22 23 24];
CHmod1(4,:) = [96 95 94 93 92 91 90 89 32 31 30 29 28 27 26 25];
CHpx1 = [88 87 86 85 36 35 34 33];
CHpx2 = [84 83 82 81 40 39 38];
CHpx3 = [80 79 78 77 44 43 42 41];
CHpx4 = [76 75 74 73 48 47 46 45];
CHpx5 = [72 71 70 69 52 51 50 49];
CHpx6 = [68 67 66 65 56 55 54 53];
CHpx7 = [64 63 62 61 60 59 58 57];

CHmod = [];
for i = 1:16
    for j = 1:4
        CHmod(end + 1) = CHmod1(j, i);
    end
end
CHpx = [CHpx1,CHpx2,CHpx3,CHpx4,CHpx5,CHpx6,CHpx7];

%% Initialize parameters

%Define the upper and lower voltage limits applied to the thermal phase shifters
V_lb = 0.5;
V_ub = 8.5;
Vpx = 4.5 + randn(1, length(CHpx));

%the outputs of two ports are combined and used as the classification result for a single port
W = zeros(10,length(label));
W(2:3,1)=100;
W(4:5,2)=100;
W(6:7,3)=100;
W(8:9,4)=100;

% Define variables to record data during the iteration process
Vpx_iteration = Vpx;
confusion_matrix_iteration=[];
mark_iteration = [];
output_iteration = [];
dv_iteration = [];

iteration = 10000;
cost0 = [];
cost1 = [];
acc_test=[];
acc_train=[];

v0_v=0;
s0_v=0;

v0_v_iteration=zeros(1,55);
s0_v_iteration=zeros(1,55);
%% Start training
for epoch = 1 : iteration
    for i = 1 : batch_num
        
        alpha_v = 0.025;
        deta_v = (rand(size(Vpx))*2 - 1)*0.025;

        cost1 = [cost1, 0];
        cost0 = [cost0, 0];
        count = 0;
        for j = 1 : batch_size_train
            
            % Load the voltage values
            load_data(data_x_train(:,j,i), CHmod, FPGA); 
            taget = find(label == data_y_train(1, j, i));

            load_data(Vpx-deta_v, CHpx, FPGA);              
            [cost_0, out, PD] = cost_function(W, taget, PD);  %Calculate the loss function
            cost0(end) = cost0(end) + cost_0; 

            output_iteration(j, :, epoch) = out;  %Record the output results
            output = ((out * W)');
            [~, m] = max(output); 
            if m == (taget)
                count = count + 1;
            end

            load_data(Vpx+deta_v, CHpx, FPGA);          
            [cost_1, ~, PD] = cost_function(W, taget, PD);
            cost1(end) = cost1(end) + cost_1;
            
        end

        cost0(end) = cost0(end) / batch_size_train; 
        cost1(end) = cost1(end) / batch_size_train; 
        acc_train = [acc_train, count / batch_size_train]; 
        
        %Calculate the test set accuracy every ten epochs
        if mod(epoch, 10)==1
            [acc, confusion_matrix, mark,PD] = predict(Vpx, W,data_x_test, data_y_test, label, CHpx, CHmod, FPGA, PD);
            confusion_matrix_iteration(:,:,end+1) = confusion_matrix;
            mark_iteration(end+1, :) = mark;
            acc_test = [acc_test, acc];
        end

        dv = deta_v * 2*(cost1(end) - cost0(end));
        dv_iteration(end+1, :) = dv; 
    
        %adam optimize
        vt_v = 0.9*v0_v + 0.1*dv;
        st_v = 0.999*s0_v + 0.001*abs(dv).^2;
        Vt_v = vt_v/(1-0.9^epoch);
        St_v = st_v/(1-0.999^epoch);
        Vpx = Vpx - alpha_v.*Vt_v./(sqrt(St_v)+10^(-9));
        v0_v = vt_v;
        s0_v = st_v;

        Vpx(Vpx > V_ub) = V_ub;
        Vpx(Vpx < V_lb) = V_lb;
        Vpx_iteration(end+1, :) = Vpx; 
        v0_v_iteration(end+1,:) = v0_v;
        s0_v_iteration(end+1,:) = s0_v;

    end    
end
%%
fclose(PD);
fclose(FPGA);
delete(instrfindall);
sprintf('All COM ports have been released.')