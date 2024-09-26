clear
clc


data_arti_1=table2array(readtable('data_xiazhang\xiazhang_1_label.csv','VariableNamingRule','preserve'))/5;

data_arti_2=table2array(readtable('data_xiazhang\xiazhang_2_label.csv','VariableNamingRule','preserve'))/5;

data_Res_3=table2array(readtable('data_xiazhang\xiazhang_3_ResNet.csv','VariableNamingRule','preserve'))/5;

data_Vit_3=table2array(readtable('data_xiazhang\xiazhang_3_Vit.csv','VariableNamingRule','preserve'))/5;

data_Res_4=table2array(readtable('data_xiazhang\xiazhang_4_ResNet.csv','VariableNamingRule','preserve'))/5;

data_Vit_4=table2array(readtable('data_xiazhang\xiazhang_4_Vit.csv','VariableNamingRule','preserve'))/5;


data_bi_arti_1=floor(data_arti_1);

[new_prediction_arti_1,event_estimate_arti_1,P_sen_like_arti_1,errorlist_1]=bayesian_fusion_no_entropy(data_bi_arti_1);

data_bi_arti_2=floor(data_arti_2);

[new_prediction_arti_2,event_estimate_arti_2,P_sen_like_arti_2,errorlist_2]=bayesian_fusion_no_entropy(data_bi_arti_2);


data_bi_Res_3=floor(data_Res_3);

[new_prediction_Res_3,event_estimate_Res_3]=predict_using_likelihood(P_sen_like_arti_1,data_bi_Res_3);

data_bi_Res_4=floor(data_Res_4);

[new_prediction_Res_4,event_estimate_Res_4]=predict_using_likelihood(P_sen_like_arti_1,data_bi_Res_4);


data_bi_Vit_3=floor(data_Vit_3);

[new_prediction_Vit_3,event_estimate_Vit_3]=predict_using_likelihood(P_sen_like_arti_1,data_bi_Vit_3);

data_bi_Vit_4=floor(data_Vit_4);

[new_prediction_Vit_4,event_estimate_Vit_4]=predict_using_likelihood(P_sen_like_arti_1,data_bi_Vit_4);


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
plot(round(mean(data_bi_Res_3)),':','linewidth',1.4)
plot(round(mean(data_bi_Vit_3)),':','linewidth',1.4)
ylim([0,1.1])
legend('ResNet+Bayesian','Vit+Bayesian','Offical','Average ResNet','Average Vit')


figure
set(gcf,'position',[100 200 800 300])

plot(event_estimate_Res_4,'--','linewidth',1.4)
hold on
plot(event_estimate_Vit_4,'-.','linewidth',1.4)

plot(data_offcial_4,'linewidth',1.4)

plot(round(mean(data_bi_Res_4)),':','linewidth',1.4)
plot(round(mean(data_bi_Vit_4)),':','linewidth',1.4)
ylim([0,1.1])
legend('ResNet+Bayesian','Vit+Bayesian','Offical','Average ResNet','Average Vit')







%% relation of interation step and error

figure

plot(errorlist_1,'-*')

title('Convergence of Bayesian fusion after classification by ResNet','fontsize',14)

xlabel('Iteration steps','fontsize',14)

ylabel('Mean squre error between steps','fontsize',14)

grid on

figure

plot(errorlist_2,'-*')

title('Convergence of Bayesian fusion after classification by Vit','fontsize',14)

xlabel('Iteration steps','fontsize',14)

ylabel('Mean squre error between steps','fontsize',14)

grid on












