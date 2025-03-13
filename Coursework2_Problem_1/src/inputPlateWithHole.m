function [nodalPositions,connectivities,DirichletBCs,NeumannBCs,mprop] = inputPlateWithHoleB(h,sigma1,sigma2)
%INPUTPLATEWITHHOLE Summary of this function goes here
%   Detailed explanation goes here
    r = 1;
    addpath('distmesh')
    geo = sprintf('ddiff(drectangle(p,0,5,0,5),dcircle(p,0,0,%0.5f))',r);
    fd = inline(geo,'p');
    fh = @(p) h + (0.5-h)/ 8. * dcircle(p,0,0,r);
    pfix = [0,5;5,5;5,0;r,0;0,r;5,0.5;5,1;5,1.5;5,2;5,2.5;5,3;5,3.5;5,4;5,4.5;5,5;0.5,5;1,5;1.5,5;2,5;2.5,5;3,5;3.5,5;4,5;4.5,5];
    bbox = [0,0;5,5];
    [nodalPositions,connectivities] = distmesh2d(fd,fh,h,bbox,pfix); 
            
    left = find(nodalPositions(:,1)<=0.0001);
    bottom = find(nodalPositions(:,2)<=0.0001);
    right = find(nodalPositions(:,1)>=(5-0.0001));
    top = find(nodalPositions(:,2)>=(5-0.0001));
    trcorner = intersect(top,right);
    tlcorner = intersect(top,left);
    brcorner = intersect(bottom,right);

    DirichletBCs = zeros(length(left)+length(bottom),3);
    for i=1:length(left)
        DirichletBCs(i,1)=left(i);
        DirichletBCs(i,2)=1;
        DirichletBCs(i,3)=0;
    end
    for i=1:length(bottom)
        DirichletBCs(i+length(left),1)=bottom(i);
        DirichletBCs(i+length(left),2)=2;
        DirichletBCs(i+length(left),3)=0;
    end
    if sigma1>0
        for i=1:length(right)
            NeumannBCs(i,1)=right(i);
            NeumannBCs(i,2)=1;
            NeumannBCs(i,3)=sigma1*0.5;
        end
    end
    if sigma2>0
        for i=1:length(top)
            NeumannBCs(i+length(right),1)=top(i);
            NeumannBCs(i+length(right),2)=2;
            NeumannBCs(i+length(right),3)=sigma2*0.5;
        end
    end
    NeumannBCs(NeumannBCs(:,1)==trcorner,3)=NeumannBCs(NeumannBCs(:,1)==trcorner,3)/2;
    NeumannBCs(NeumannBCs(:,1)==tlcorner,3)=NeumannBCs(NeumannBCs(:,1)==tlcorner,3)/2;
    NeumannBCs(NeumannBCs(:,1)==brcorner,3)=NeumannBCs(NeumannBCs(:,1)==brcorner,3)/2;
    
    mprop = [ 10 0.3 1.0]; % [E, v , t ] 
    connectivities(:,4) = 1;
end

