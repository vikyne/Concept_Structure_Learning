
% open required dataset

% Bayesian model reduction
%--------------------------------------------------------------------------
OPTIONS.g = 3;
% OPTIONS.f = 2;
% OPTIONS.T = 3;
% OPTIONS.m = @(i,i1,i2,i3,i4) i == i3;

N = numel(MMDP);

sdp{N} = spm_MDP_VB_sleep_16_alt(MMDP(N).mdp(end),OPTIONS);

for l = 1:16

    spm_figure('GetWin','Figure'); clf; str = sprintf('Sleep (trial %d)',N);
    subplot(3,2,1), imagesc(spm_norm(MMDP(N).mdp(end).A{3}(:,:,l)));
    subplot(3,2,2), imagesc(MMDP(N).mdp(end).a0{3}(:,:,l)), title('Before','Fontsize',16)
    colorbar
    subplot(3,2,3), imagesc(spm_norm(MMDP(N).mdp(end).a{3}(:,:,l))),  title('After', 'Fontsize',16)
    subplot(3,2,4), imagesc(spm_norm(sdp{N}.a{3}(:,:,l))),  title(str,     'Fontsize',16)
    hold on
    colormap('summer')
    
end
BMR_a = sdp{N}.a;
% save BMR_ag_x BMR_a

