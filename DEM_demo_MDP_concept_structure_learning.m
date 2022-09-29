
% This is an active inference routine of spatial foraging using a discrete
% state space MDP

clear all

N = 10;

% First level L1

% Rooms have identical size (4x4). In each room start from left and down
% of centre. Each room has a unique reward location, but 2 sets of rooms
% are identical. Agent explores the grid room where the room is one of 16

d{1} = ones(16,1);   % context: room identity 
d{2} = zeros(16,1);  % location:  there are 16 locations
d{2}(7) = 1;         % start from location 7

Nf    = numel(d);
for f = 1:Nf
    Ns(f) = numel(d{f});
end

No    = [16 2 16]; % {16 locations}, { null, reward}, {16 contextual cues}, 
                   % cueing what room agent is in - e.g. room colour 
Ng    = numel(No);
for g = 1:Ng
    A{g} = zeros([No(g),Ns]);
end

for f1 = 1:Ns(1) % context: room identity s
    for f2 = 1:Ns(2) % location: one of 16
        
        if f1 == 1 % starting room rew at 10
            Room = {'null' 'null' 'null' 'null'; 'null' 'null' 'null' 'null'; 'null' 'rew' 'null' 'null'; 'null' 'null' 'null' 'null'}';
        elseif f1 == 2 % rew at 1
            Room = {'rew' 'null' 'null' 'null'; 'null' 'null' 'null' 'null'; 'null' 'null' 'null' 'null'; 'null' 'null' 'null' 'null'}';
        elseif f1 == 3  % rew at 9
            Room = {'null' 'null' 'null' 'null'; 'null' 'null' 'null' 'null'; 'rew' 'null' 'null' 'null'; 'null' 'null' 'null' 'null'}';
        elseif f1 == 4 % rew at 14
            Room = {'null' 'null' 'null' 'null'; 'null' 'null' 'null' 'null'; 'null' 'null' 'null' 'null'; 'null' 'rew' 'null' 'null'}';
        elseif f1 == 5 % rew at 5
            Room = {'null' 'null' 'null' 'null'; 'rew' 'null' 'null' 'null'; 'null' 'null' 'null' 'null'; 'null' 'null' 'null' 'null'}';
        elseif f1 == 6 % rew at 2
            Room = {'null' 'rew' 'null' 'null'; 'null' 'null' 'null' 'null'; 'null' 'null' 'null' 'null'; 'null' 'null' 'null' 'null'}';
        elseif f1 == 7 % rew at 13
            Room = {'null' 'null' 'null' 'null'; 'null' 'null' 'null' 'null'; 'null' 'null' 'null' 'null'; 'rew' 'null' 'null' 'null'}';
        elseif f1 == 8 % rew at 11
            Room = {'null' 'null' 'null' 'null'; 'null' 'null' 'null' 'null'; 'null' 'null' 'rew' 'null'; 'null' 'null' 'null' 'null'}';
        elseif f1 == 9 % rew at 3
            Room = {'null' 'null' 'rew' 'null'; 'null' 'null' 'null' 'null'; 'null' 'null' 'null' 'null'; 'null' 'null' 'null' 'null'}';
        elseif f1 == 10  % rew at 12
            Room = {'null' 'null' 'null' 'null'; 'null' 'null' 'null' 'null'; 'null' 'null' 'null' 'rew'; 'null' 'null' 'null' 'null'}';
        elseif f1 == 11 % rew at 4
            Room = {'null' 'null' 'null' 'rew'; 'null' 'null' 'null' 'null'; 'null' 'null' 'null' 'null'; 'null' 'null' 'null' 'null'}';
        elseif f1 == 12 % rew at 16
            Room = {'null' 'null' 'null' 'null'; 'null' 'null' 'null' 'null'; 'null' 'null' 'null' 'null'; 'null' 'null' 'null' 'rew'}';
        elseif f1 == 13 % rew at 15
            Room = {'null' 'null' 'null' 'null'; 'null' 'null' 'null' 'null'; 'null' 'null' 'null' 'null'; 'null' 'null' 'rew' 'null'}';
        elseif f1 == 14 % rew at 6
            Room = {'null' 'null' 'null' 'null'; 'null' 'rew' 'null' 'null'; 'null' 'null' 'null' 'null'; 'null' 'null' 'null' 'null'}';
        elseif f1 == 15 % rew at 1 - alike room 2
            Room = {'rew' 'null' 'null' 'null'; 'null' 'null' 'null' 'null'; 'null' 'null' 'null' 'null'; 'null' 'null' 'null' 'null'}';
        elseif f1 == 16 % rew at 9 - alike room 3
            Room = {'null' 'null' 'null' 'null'; 'null' 'null' 'null' 'null'; 'rew' 'null' 'null' 'null'; 'null' 'null' 'null' 'null'}';
        end
        
        A{1}(f2,f1,f2) = 1; % A{outcome modality}
                            % (location outcome,context, location state)
                            % mapping from location to location regardless
                            % of context         
        
        A{2}(1,f1,f2) = strcmp(Room{f2},'null'); % no-reward
        A{2}(2,f1,f2) = strcmp(Room{f2},'rew');  % reward
    
        A{3}(f1,f1,f2) = 1; % (colour, context cue, room identity)
                            % mapping from room type to itself i.e. each
                            % room has one color
    end
end

for g = 1:Ng
    A{g} = double(A{g});
    a{g} = A{g};                                 
end

a{1} = 512*A{1};
a{2} = power(10,-1)*A{2};
a{3} = power(10,-1)*ones(size(A{3}));


% controlled transitions: B{f} for each factor
%--------------------------------------------------------------------------
for f = 1:Nf
    B{f} = eye(Ns(f));
end

% controlled transitions: B (down, up, right, left)
%--------------------------------------------------------------------------
u    = [1 0; -1 0; 0 1; 0 -1];               % allowable actions
nu   = size(u,1);                            % number of actions
B{2} = zeros(Ns(2),Ns(2),nu);
[n,m] = size(Room);
for i = 1:n
    for j = 1:m
        
        % allowable transitions from state s to state ss
        %------------------------------------------------------------------
        s     = sub2ind([n,m],i,j);
        for k = 1:nu
            try
                ss = sub2ind([n,m],i + u(k,1),j + u(k,2));
                B{2}(ss,s,k) = 1;
            catch
                B{2}(s, s,k) = 1;
            end
        end
    end
end


% allowable policies (4 moves): V
%--------------------------------------------------------------------------
V     = [];
for i = 1:nu
    for j = 1:nu
        for k = 1:nu
            for l = 1:nu
                V(:,end + 1,2) = [i;j;k;l]; 
            end    
        end 
    end
end

V(find(V==0)) = 1; % context doesn't change 

T = 5; % number of time steps on the lower level 

C{1} = zeros(No(1),T); % no preferences over locations
C{2} = zeros(No(2),T); % preferences over rewards (see below)
C{3} = zeros(No(3),T); % no preferences over colours

C{2}(2,:) = 3; % positive utility for reward
C{2}(:,6:end) = -100; 

% MDP Structure
%--------------------------------------------------------------------------
mdp.T = T;                      % number of updates
mdp.a = a;                      % 1st level likelihood conc. parameters
mdp.a0 = a;                     % observation model
mdp.A = A;                      % observation model
mdp.B = B;                      % transition probabilities
mdp.C = C;                      % prior over preferences
mdp.D = d;                      % prior over initial states
mdp.V = V;                      % allowable policies

mdp.chi = -30;                  % occam's threshold


mdp.Aname = {'Location','Feedback','Contextual Cue'};
mdp.Bname = {'Room identity','Location'};
clear A B D
 
MDP = spm_MDP_check(mdp);
clearvars -except MDP N


% Second level L2

% The agent starts from room 1. It moves between four other different 
% rooms for every iteration

% prior beliefs about initial states (in terms of counts)
%--------------------------------------------------------------------------
d{1} = ones(16,1); % context (hidden state); 


% probabilistic mapping from hidden states to outcomes: A
%--------------------------------------------------------------------------
Nf    = numel(d); 
for f = 1:Nf 
    Ns(f) = numel(d{f}); 
end
No    = [16]; % outcome modality: type of room i.e., colour 
A{1} = eye(16);


% controllable fixation points: move to the k-th location
%--------------------------------------------------------------------------
for k = 1:Ns(1) 
   B{1}(:,:,k) = zeros(Ns(1),Ns(1)); 
   B{1}(k,:,k) = ones(1,16);
end  
 
% allowable policies (here, specified as the next action) U
%--------------------------------------------------------------------------
Np = size(B{1},3); % number of actions
U  = ones(1,Np,Nf);
U(:,:,1)  = 1:Np; % number of policies 
 
% priors: (utility) C
%--------------------------------------------------------------------------
T         = 5; 
C{1}      = zeros(No(1),T); % no prior preferences on this level


% MDP Structure
%--------------------------------------------------------------------------
mdp.MDP  = MDP;
mdp.link = [1; 0]; % outcome of L2 enter at L1 as initial state for context 

% MDP Structure - this will be used to generate arrays for multiple trials
%==========================================================================
mdp.T = T;                      % number of moves
mdp.U = U;                      % allowable actions
mdp.A = A;                      % observation model
mdp.B = B;                      % transition probabilities
mdp.C = C;                      % preferred outcomes
mdp.D = d;                      % prior over initial states
mdp.s = [1]';                   % initial states 

mdp.Aname = {'Room identity'}; 
mdp.Bname = {'Context'}; 
 
mdp = spm_MDP_check(mdp);

% illustrate a single trial
%==========================================================================

BeforeSim = mdp; % before simulations
for i = 1:N
    MMDP(i)   = spm_MDP_VB_X(mdp);
    
    for j = 1:numel(mdp.MDP.a)
        mdp.MDP.a{j} = MMDP(i).mdp(end).a{j};
    end
    
end 
AfterSim = mdp; % after simulations


% save agent_x BeforeSim AfterSim MMDP

