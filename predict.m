function [acc,confusion_matrix,mark,PD] = predict(Vpx, W,data_x, data_y, label, CHpx, CHmod, FPGA, PD,)
    
    data_size = size(data_x);
    if length(data_size) == 2
        data_size(3) = 1;
    end
    count = 0;
    confusion_matrix = zeros(length(label), length(label));
    mark = zeros(1, length(data_y));
    load_data(Vpx, CHpx, FPGA);
    
    for i = 1: data_size(3)
        for j = 1: data_size(2)            
            load_data(data_x(:,j,i), CHmod, FPGA);
            taget = find(label==data_y(1, j, i));
            [~, output, PD] = cost_function4(W, taget, PD);    
            output = softmax(abs(output * W)');
        
            [~, m] = max(output);   
            mark(j) = m;
            if m == (taget)
                count = count + 1;
            end
            confusion_matrix(taget, m) =  confusion_matrix(taget, m) + 1;
        end
    end

    acc = count / (data_size(2) * data_size(3));
end 