function [new_prediction,event_estimate]=predict_using_likelihood(P_sen_like,data_bi)

[N_sen,N_t]=size(data_bi);



P_z1=sum(data_bi(:))/N_sen/N_t;

P_z0=1-P_z1;


P_z1_con_Y=ones(1,N_t);

P_z0_con_Y=ones(1,N_t);

P_sen_like_all_1=ones(N_sen,N_t);

P_sen_like_all_0=ones(N_sen,N_t);

for j=1:N_t

    for k=1:N_sen

        P_sen_like_all_1(k,j)=P_sen_like(k,(2-data_bi(k,j)));

        P_sen_like_all_0(k,j)=P_sen_like(k,(4-data_bi(k,j)));

    end

    P_z1_con_Y(j)=(prod(P_sen_like_all_1(:,j))*P_z1)/...
        (prod(P_sen_like_all_1(:,j))*P_z1+prod(P_sen_like_all_0(:,j))*P_z0);
    P_z0_con_Y(j)=1-P_z1_con_Y(j);
    


end


event_estimate=round(P_z1_con_Y);

new_prediction=kron(ones(N_sen,1),event_estimate);

end