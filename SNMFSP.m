function [W,H,noiter,obj]=SNMFSP(X,W,H,LS,M,B,E,C,Clabel,options)
% Semi-supervised non-negative matrix factorization with structure preserving for image clustering
% X Data matrix 
% W Basis matrix
% H Encoding matrix
% LS Weigth matrix
% M Weighted label matrix
% B Basis matrix of labeled data
% E Fitting matrix
% C Consist of the Clabel-th rows of H and 0 vectors
% Clabel Number of labeled samples 

eps=options.eps;
alpha=options.alpha;
lambda=options.lambda;
Maxiter=options.Maxiter;
LD=sum(LS,2);
LD=diag(LD);
%L=LD-LS;

for k=1:Maxiter
    %% 更新W
    WFZ=X*H+1*E*B;
    WFZ(WFZ<0)=0;
    WFM=W*H'*H+1*W;
    W=W.*(WFZ./max(WFM,1e-15));
    
    %% 更新H;
    obj=H;
    XTW=X'*W; 
    HFZ=XTW+alpha*M'*B+lambda*LS*H;
    HFZ(HFZ<0)=0;
    HWTW=H*W'*W;
    HFM=HWTW+alpha*C*B'*B+lambda*LD*H;
    H=H.*(HFZ./max(HFM,1e-15));
    
    C(Clabel,:)=H(Clabel,:);
   %% 更新B
    BFZ=alpha*M*C+1*E'*W;
    BFZ(BFZ<0)=0;
    BFM=alpha*B*C'*C+1*E'*E*B;
    B=B.*BFZ./max(BFM,1e-15);
    
    %% 更新E
    EFZ=W*B';
    EFZ(EFZ<0)=0;
    EFM=E*B*B';
    E=E.*EFZ./max(EFM,1e-15); 
    
    %% 
    obj0=H;
    re=norm(obj0-obj)/norm(obj);
    if re<eps
       break
    end        
   
end
     noiter=k;
end