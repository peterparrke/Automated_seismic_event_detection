function [new_prediction,event_estimate,P_sen_like]=max_entropy_bayesian(data_bi)
[N_sen,N_t]=size(data_bi);

event_mean=mean(data_bi);

P_z1=sum(data_bi(:))/N_sen/N_t;

P_z0=1-P_z1;

P_sen_like=zeros(N_sen,4);% matrix of likelihood of each sensor
                                             % 1st column is P(1|True=1),
                                             % 2nd is P(0|True=1),
                                             % 3rd is P(1|True=0),
                                             % 4th is P(0|True=0).

event_num=ceil(P_z1*N_t);
event_estimate=zeros(1,N_t);

[~,sort_event_chance]=sort(event_mean,'descend');

pos_is_event=sort_event_chance(1:event_num);

pos_not_event=setdiff([1:1:N_t],pos_is_event);

event_estimate(pos_is_event)=1;

% figure
% plot(event_estimate,'--')
% hold on

for j=1:N_sen
    
    P_sen_like(j,1)=1 /N_t/2+sum(data_bi(j,pos_is_event))/sum(event_estimate);
    
    P_sen_like(j,2)=1/N_t+1-P_sen_like(j,1);
    
    P_sen_like(j,3)=1/N_t/2+sum(data_bi(j,pos_not_event))/(N_t-sum(event_estimate));
    
    P_sen_like(j,4)=1/N_t+1-P_sen_like(j,3);
    
end

P_sen_like=P_sen_like/(1/N_t+1);

P_z1_con_Y=ones(1,N_t);

P_z0_con_Y=ones(1,N_t);

P_sen_like_all_1=ones(N_sen,N_t);

P_sen_like_all_0=ones(N_sen,N_t);

H_new=0;
for j=1:N_t
    
    for k=1:N_sen
        
        P_sen_like_all_1(k,j)=P_sen_like(k,(2-data_bi(k,j)));
        
        P_sen_like_all_0(k,j)=P_sen_like(k,(4-data_bi(k,j)));
        
    end
    
    P_z1_con_Y(j)=(prod(P_sen_like_all_1(:,j))*P_z1)/...
        (prod(P_sen_like_all_1(:,j))*P_z1+prod(P_sen_like_all_0(:,j))*P_z0);
    P_z0_con_Y(j)=1-P_z1_con_Y(j);
    
    H_new=H_new+prod(P_sen_like_all_1(:,j))*P_z1*log(prod(P_sen_like_all_1(:,j))*P_z1/...
        (prod(P_sen_like_all_1(:,j))*P_z1+prod(P_sen_like_all_0(:,j))*P_z0));
    H_new=H_new+prod(P_sen_like_all_0(:,j))*P_z0*log(prod(P_sen_like_all_0(:,j))*P_z0/...
        (prod(P_sen_like_all_1(:,j))*P_z1+prod(P_sen_like_all_0(:,j))*P_z0));% entropy
    
end

    







%%
flag=1;
cnt=0;

d_H=1;


while flag

    event_estimate=round(P_z1_con_Y);
    
%     plot(P_z1_con_Y,'--')
%     hold on
    
    P_z1=sum(event_estimate)/N_t;
    P_z0=1-P_z1;
    pos_is_event=find(event_estimate==1);
    pos_not_event=setdiff([1:1:N_t],pos_is_event);
    

    for j=1:N_sen
    
        P_sen_like(j,1)=1/N_t/2+sum(data_bi(j,pos_is_event))/sum(event_estimate);

        P_sen_like(j,2)=1/N_t+1-P_sen_like(j,1);

        P_sen_like(j,3)=1/N_t/2+sum(data_bi(j,pos_not_event))/(N_t-sum(event_estimate));

        P_sen_like(j,4)=1/N_t+1-P_sen_like(j,3);

    end

    P_sen_like=P_sen_like/(1/N_t+1);

    P_z1_con_Y=ones(1,N_t);

    P_z0_con_Y=ones(1,N_t);

    P_sen_like_all_1=ones(N_sen,N_t);

    P_sen_like_all_0=ones(N_sen,N_t);
    
    H_old=H_new;

    H_new=0;
    for j=1:N_t

        for k=1:N_sen

            P_sen_like_all_1(k,j)=P_sen_like(k,(2-data_bi(k,j)));

            P_sen_like_all_0(k,j)=P_sen_like(k,(4-data_bi(k,j)));

        end

        P_z1_con_Y(j)=(prod(P_sen_like_all_1(:,j))*P_z1)/...
            (prod(P_sen_like_all_1(:,j))*P_z1+prod(P_sen_like_all_0(:,j))*P_z0);
        P_z0_con_Y(j)=1-P_z1_con_Y(j);
        
        H_new=H_new+prod(P_sen_like_all_1(:,j))*P_z1*log(prod(P_sen_like_all_1(:,j))*P_z1/...
            (prod(P_sen_like_all_1(:,j))*P_z1+prod(P_sen_like_all_0(:,j))*P_z0));
        H_new=H_new+prod(P_sen_like_all_0(:,j))*P_z0*log(prod(P_sen_like_all_0(:,j))*P_z0/...
            (prod(P_sen_like_all_1(:,j))*P_z1+prod(P_sen_like_all_0(:,j))*P_z0));% entropy

    end
    
    d_H_old=d_H;
    
    d_H=H_new-H_old;
    
    
    
    cnt=cnt+1;
    
    if cnt>=20 || d_H_old*d_H<=0
        flag=0;
    end
    
end


new_prediction=kron(ones(N_sen,1),event_estimate);

end