% load leleccum;
% index=1:1024;
% f1=leleccum(index); % ���������ź�
inputfile=xlsread('D:/2015b/bin/����ǰһʱ������.xlsx');
f1=inputfile(:,6);
f2=f1';
% init=2055615866;
% randn('seed',init);
% f2=f1+18*randn(size(f1));
% snr=SNR_singlech(f1,f2) %�����
% subplot(2,2,1);plot(f1);title('�����ź�'); %axis([1,1024,-1,1]);
subplot(2,1,1);
x=[1:1433];
y=f2;
p=find(y==min(y));
% plot(x(p),output_test(p),'g*')
% x=[1:286];
text(x(p),f2(p),'*','color','g');
index=find(x==741);
annotation('textarrow',[0.49 0.425],[0.78 0.705],'String',['x = ',num2str(x(index)),',y = ',num2str(y(index))],'color','k');
plot(f2);title('Noisy Signal'); %axis([1,1024,-1,1]);
xlabel('Number');ylabel('Permeability Index');
set(gca,'FontSize',12,'FontName','Times New Roman','linewidth',1.5,'FontWeight','bold');
%��db5С����ԭʼ�źŽ���3��ֽⲢ��ȡϵ��
[c,l]=wavedec(f2,3,'db6');
a3=appcoef(c,l,'db6',3);
d3=detcoef(c,l,3);
d2=detcoef(c,l,2);
d1=detcoef(c,l,1);
sigma=wnoisest(c,l,1);
thr=wbmpen(c,l,sigma,2);
%����Ӳ��ֵ����
ythard1=wthresh(d1,'h',thr);
ythard2=wthresh(d2,'h',thr);
ythard3=wthresh(d3,'h',thr);
c2=[a3 ythard3 ythard2 ythard1];
f3=waverec(c2,l,'db6');
%��������ֵ����
ytsoftd1=wthresh(d1,'s',thr);
ytsoftd2=wthresh(d2,'s',thr);
ytsoftd3=wthresh(d3,'s',thr);
c3=[a3 ytsoftd3 ytsoftd2 ytsoftd1];
f4=waverec(c3,l,'db6');
%�������źŽ���ͼʾ
% subplot(2,2,3);plot(f3);title('Ӳ��ֵ����');%axis([1,1024,-1,1]);
subplot(2,1,2);
x=[1:1433];
y=f4;
p=find(y==min(y));
% plot(x(p),output_test(p),'g*')
% x=[1:286];
text(x(p),f4(p),'*','color','g');
index=find(x==741);
annotation('textarrow',[0.49 0.425],[0.78 0.705],'String',['x = ',num2str(x(index)),',y = ',num2str(y(index))],'color','k');
plot(f4);title('Wavelet De-noising');%axis([1,1024,-1,1]);
xlabel('Number')
ylabel('Permeability Index');
set(gca,'FontSize',12,'FontName','Times New Roman','linewidth',1.5,'FontWeight','bold');
% snr=SNR_singlech(f3,f2)
% snr=SNR_singlech(f4,f2)