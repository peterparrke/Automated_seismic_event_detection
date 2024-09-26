clear
clc

data_Res_3=table2array(readtable('data_xiazhang\xiazhang_3_ResNet.csv','VariableNamingRule','preserve'))/5;

data_Vit_3=table2array(readtable('data_xiazhang\xiazhang_3_Vit.csv','VariableNamingRule','preserve'))/5;

data_Res_4=table2array(readtable('data_xiazhang\xiazhang_4_ResNet.csv','VariableNamingRule','preserve'))/5;

data_Vit_4=table2array(readtable('data_xiazhang\xiazhang_4_Vit.csv','VariableNamingRule','preserve'))/5;


data_bi_Res_3=floor(data_Res_3);

[new_prediction_Res_3,event_estimate_Res_3,P_sen_like_Res_3]=max_entropy_bayesian(data_bi_Res_3);

data_bi_Res_4=floor(data_Res_4);

[new_prediction_Res_4,event_estimate_Res_4,P_sen_like_Res_4]=max_entropy_bayesian(data_bi_Res_4);

data_bi_Vit_3=floor(data_Vit_3);

[new_prediction_Vit_3,event_estimate_Vit_3,P_sen_like_Vit_3]=max_entropy_bayesian(data_bi_Vit_3);

data_bi_Vit_4=floor(data_Vit_4);

[new_prediction_Vit_4,event_estimate_Vit_4,P_sen_like_Vit_4]=max_entropy_bayesian(data_bi_Vit_4);


data_offcial_3=importdata('data_xiazhang\xiazhang_3.dat');

data_offcial_4=importdata('data_xiazhang\xiazhang_4.dat');


%%
figure
set(gcf,'position',[100 200 800 300])
plot(event_estimate_Res_3,'--','linewidth',1.4)
hold on
plot(event_estimate_Vit_3,'-.','linewidth',1.4)
hold on
plot(data_offcial_3,'linewidth',1.4)
ylim([0,1.1])
legend('ResNet','Vit','Offical')


figure
set(gcf,'position',[100 200 800 300])

plot(event_estimate_Res_4,'--','linewidth',1.4)
hold on
plot(event_estimate_Vit_4,'-.','linewidth',1.4)

plot(data_offcial_4,'linewidth',1.4)
ylim([0,1.1])
legend('ResNet','Vit','Offical')

















