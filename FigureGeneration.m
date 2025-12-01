load("Data.mat")
load("NPG.mat")
figure;
loglog((mean(chistoryGN,2)-trace(Pstar)*ones(iteration,1))/trace(Pstar),'LineStyle','-','Color','red','LineWidth',2)
hold on
loglog((mean(chistoryGNL3,2)-trace(Pstar)*ones(iteration,1))/trace(Pstar),'LineStyle','-.','Color','black','LineWidth',2)
hold on
loglog((mean(chistoryGNL2,2)-trace(Pstar)*ones(iteration,1))/trace(Pstar),'LineStyle',':','Color','blue','LineWidth',2)
hold on
legend("GNM CSPD (Algo [2])$","GNM CSPD (Algo [3])","GNM LS estimating $\hat{A},\hat{B}$",'interpreter','latex','FontSize',18)
ylabel('$\frac{C{(\hat{K}_i)}-C{(K^*)}}{C{(K^*)}}$','interpreter','latex','FontSize',18)
xlabel('$\mathrm{iteration}~i$','interpreter','latex','FontSize',18)
grid on
ylim([0.05,3])
xlim([1,35])

figure;
loglog((mean(chistoryNP,2)-trace(Pstar)*ones(iteration,1))/trace(Pstar),'LineStyle','-','Color','red','LineWidth',2)
hold on
loglog((mean(chistoryNPL,2)-trace(Pstar)*ones(iteration,1))/trace(Pstar),'LineStyle','--','Color','black','LineWidth',2)
hold on
loglog((mean(ckBS,2)-trace(Pstar)*ones(110,1))/trace(Pstar),'LineStyle',':','Color','blue','LineWidth',2)
legend("NPG CSPD (Algo [2])","NPG LS using (18)","NPG Zeroth-order (Song et al. 2025a)",'interpreter','latex','FontSize',18)
ylabel('$\frac{C{(\hat{K}_i)}-C{(K^*)}}{C{(K^*)}}$','interpreter','latex','FontSize',18)
xlabel('$\mathrm{iteration}~i$','interpreter','latex','FontSize',18)
grid on
ylim([0.05,3])
xlim([1,35])
