function [cost, output, PD] = cost_function(W , taget, PD)
    
    [output, PD]= PD_readoutall_10(PD);  % get the value of the output
    output = 10.^(output/10); % Convert the unit from dBm to mW
    output1 = softmax(abs(output * W)');
    cost = - log(output1(taget)); %Calculate the loss function
    
end

