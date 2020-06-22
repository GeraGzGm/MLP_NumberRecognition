clc;clear;
Images = loadMNISTImages('train-images.idx3-ubyte');
Images = reshape(Images,28,28,[]);
Labels = loadMNISTLabels('train-labels.idx1-ubyte');
Labels(Labels == 0) = 10; %0 --> 10 

imagenes ='si'; 

X = Images(:,:,1:8000);
d = Labels(1:8000);

nh = 25;
no = 10;

wh = -2+(2+2)*rand(nh,784);
bh = -2+(2+2)*rand(nh,1);
wo = -2+(2+2)*rand(no,nh);
bo = -2+(2+2)*rand(no,1);

N = 0.01;

for ep = 1:5000
    sum_ = 0;
    perm = randperm(length(X));
    for i=1:length(perm)*.90
        inp = reshape(X(:,:,perm(i)),[],1);
        
        h = wh*inp+bh;
        yh = tg_sig(h);
        
        o = wo*yh+bo;
        yo = logistic_(o);
        

        lel = zeros(10,1);
        lel(d(perm(i))) = 1;  
        e = lel-yo;
        
        do = -2*dlogistic_(yo).*e;
        dh = dtanh_(yh).*(wo'*do);
        
        wo = wo - N*do*yh';
        bo = bo - N*do;
        
        wh = wh - N*dh*inp';
        bh = bh - N*dh; 
        
        sum_ = sum_+e;
    end
    et= sum_/length(perm)*.90;
end

d(d==10) = 0;
for i=length(perm)*.90+1:length(perm)
        inp = reshape(X(:,:,perm(i)),[],1);
        
        h = wh*inp+bh;
        yh = tg_sig(h);
        o = wo*yh+bo;
        yo = tg_sig(o);
        
        
        
        [~,cual] = max(yo);
        
        cual(cual==10)=0;
        
        if imagenes == 'si'
         imshow(reshape(inp,28,28));
         title(['Salida de la red:',num2str(cual),'  Deseada:',num2str(d(perm(i)))]);
          pause();
        end
        
        red(i-length(perm)*.90+1) = cual;
        des(i-length(perm)*.90+1) = d(perm(i));
end

NumeroAciertos = sum(red==des);
Porcentaje  = NumeroAciertos/length(red)*100

function y = tg_sig(x)
    y = (exp(x) - exp(-x))./(exp(x) + exp(-x));
end

    function y = dtanh_(x)
        y = 1-x.^2;
    end
    
    function y = logistic_(x)
        y = 1./(1+exp(-x));
   end
    
    function y = dlogistic_(x)
        y = x.*(1-x);
    end
    
