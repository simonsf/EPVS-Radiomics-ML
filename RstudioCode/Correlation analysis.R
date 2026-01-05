# 相关性热图

# 若没有需要先安装对应d包
#install.packages("ISLR2")
library(corrplot) # 这个包是今天可视化的主角
library(RColorBrewer) # 用来配色的
library(ggcorrplot2)
library(ggforce)

data <- read.csv("M:/52_Chengdu_Lili/文章2/Correlation/Correlation analysis.csv")
tr_data <- data[data$Group == "Train", ]
ts_data <- data[data$Group == "Test", ]

new_data = tr_data[, c(14,13,12,9,10,11)]

# 使用corrplot包计算相关性
cor_matrix = cor(new_data, method = "spearman", use = "pairwise.complete.obs")

# 计算相关性的P值和置信区间
pData = cor.mtest(new_data, conf.level = 0.95, method = "spearman")
p_matrix <- pData$p

# 导出8.56 * 4.11
ggcorrplot.mixed(cor_matrix, upper = "ellipse", lower = "number",
                 col = colorRampPalette(c("navy", "white", "firebrick3"))(200),
                 p.mat = p_matrix, sig.lvl = 0.05,
                 number.digits = 2, pch = 4, pch.cex = 10,
                 insig = "label_sig")

