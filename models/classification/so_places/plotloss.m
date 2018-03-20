clear;
clc;
close all;
train_log_file='Googlelog';
train_interval=20;
%test_interval=1;
[~, stringbbox_loss]=dos(['cat ', train_log_file, ' | grep ''Train net output #0: prob_offset'' | awk ''{print $11}'' ']);
train_lossbbox_loss=str2num(stringbbox_loss);

[~, stringcls_loss]=dos(['cat ', train_log_file, ' | grep ''Train net output #1: prob_slope'' | awk ''{print $11}'' ']);
train_losscls_loss=str2num(stringcls_loss);


n=1:length(train_lossbbox_loss);
idx_trainfast=(n-1)*train_interval;

figure
plot(idx_trainfast,train_lossbbox_loss);
hold on
plot(idx_trainfast,train_losscls_loss);
legend('offset','slope');



%[~, stringrpnbbox_loss]=dos(['cat ', train_log_file, ' | grep ''Train net output #1: rpn_loss_bbox'' | awk ''{print $11}'' ']);
%train_loss_rpn_bbox_loss=str2num(stringrpnbbox_loss);

%[~, stringrpncls_loss]=dos(['cat ', train_log_file, ' | grep ''Train net output #0: rpn_cls_loss'' | awk ''{print $11}'' ']);
%train_loss_rpn_cls_loss=str2num(stringrpncls_loss);


%n=1:length(train_loss_rpn_bbox_loss);
%idx_train_rpn_fast=(n-1)*train_interval;

%figure
%plot(idx_train_rpn_fast,train_loss_rpn_bbox_loss);
%hold on
%plot(idx_train_rpn_fast,train_loss_rpn_cls_loss);
%legend('rpn bbox loss','rpn cls loss');
