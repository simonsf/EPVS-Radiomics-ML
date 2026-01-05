#绘制混淆矩阵---------------------------------------------------------------
rm(list = ls())
library(caret)
library(ggplot2)

label <- factor(c(0, 0, 1, 1))
pred <- factor(c(0, 1, 0, 1))

# ISI_TRAINING---------------------------
n <- c(34, 7, 2, 22)
df <- data.frame(label, pred, n)
ggplot(data = df, mapping = aes(x = label, y = pred)) +
  geom_tile(aes(fill = n), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f", n)), vjust = 1) +
  scale_fill_gradient(low = "ghostwhite", high = "#7F89B8", space = "Lab") +
  theme_bw() +
  theme(legend.position = "none") +
  scale_y_discrete(limits=rev) +
  xlab("Predicted ISI category") +
  ylab("Actual ISI category")
ggsave(filename = ".../Confusion matrix/ISI_training.pdf",
       plot = last_plot(),
       width = 2.4, height = 2.4, unit = "in", dpi = 300)

# ISI_TESTING
n <- c(8, 2, 2, 5)
df <- data.frame(label, pred, n)
ggplot(data = df, mapping = aes(x = label, y = pred)) +
  geom_tile(aes(fill = n), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f", n)), vjust = 1) +
  scale_fill_gradient(low = "ghostwhite", high = "#7F89B8", space = "Lab") +
  theme_bw() +
  theme(legend.position = "none") +
  scale_y_discrete(limits=rev) +
  xlab("Predicted ISI category") +
  ylab("Actual ISI category")
ggsave(filename = ".../Confusion matrix/ISI_testing.pdf",
       plot = last_plot(),
       width = 2.4, height = 2.4, unit = "in", dpi = 300)

# MOCA_TRAINING---------------------------
n <- c(19, 6, 2, 38)
df <- data.frame(label, pred, n)
ggplot(data = df, mapping = aes(x = label, y = pred)) +
  geom_tile(aes(fill = n), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f", n)), vjust = 1) +
  scale_fill_gradient(low = "ghostwhite", high = "#7F89B8", space = "Lab") +
  theme_bw() +
  theme(legend.position = "none") +
  scale_y_discrete(limits=rev) +
  xlab("Predicted MoCA category") +
  ylab("Actual MoCA category")
ggsave(filename = ".../Confusion matrix/MOCA_training.pdf",
       plot = last_plot(),
       width = 2.4, height = 2.4, unit = "in", dpi = 300)

# MOCA_TESTING
n <- c(4, 3, 2, 8)
df <- data.frame(label, pred, n)
ggplot(data = df, mapping = aes(x = label, y = pred)) +
  geom_tile(aes(fill = n), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f", n)), vjust = 1) +
  scale_fill_gradient(low = "ghostwhite", high = "#7F89B8", space = "Lab") +
  theme_bw() +
  theme(legend.position = "none") +
  scale_y_discrete(limits=rev) +
  xlab("Predicted MoCA category") +
  ylab("Actual MoCA category")
ggsave(filename = ".../Confusion matrix/MOCA_testing.pdf",
       plot = last_plot(),
       width = 2.4, height = 2.4, unit = "in", dpi = 300)

# PSQI_TRAINING---------------------------
n <- c(16, 4, 6, 39)
df <- data.frame(label, pred, n)
ggplot(data = df, mapping = aes(x = label, y = pred)) +
  geom_tile(aes(fill = n), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f", n)), vjust = 1) +
  scale_fill_gradient(low = "ghostwhite", high = "#7F89B8", space = "Lab") +
  theme_bw() +
  theme(legend.position = "none") +
  scale_y_discrete(limits=rev) +
  xlab("Predicted PSQI category") +
  ylab("Actual PSQI category")
ggsave(filename = ".../Confusion matrix/PSQI_training.pdf",
       plot = last_plot(),
       width = 2.4, height = 2.4, unit = "in", dpi = 300)

# PSQI_TESTING
n <- c(4, 1, 2, 10)
df <- data.frame(label, pred, n)
ggplot(data = df, mapping = aes(x = label, y = pred)) +
  geom_tile(aes(fill = n), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f", n)), vjust = 1) +
  scale_fill_gradient(low = "ghostwhite", high = "#7F89B8", space = "Lab") +
  theme_bw() +
  theme(legend.position = "none") +
  scale_y_discrete(limits=rev) +
  xlab("Predicted PSQI category") +
  ylab("Actual PSQI category")
ggsave(filename = ".../Confusion matrix/PSQI_testing.pdf",
       plot = last_plot(),
       width = 2.4, height = 2.4, unit = "in", dpi = 300)

# ESS_TRAINING---------------------------
n <- c(33, 8, 0, 24)
df <- data.frame(label, pred, n)
ggplot(data = df, mapping = aes(x = label, y = pred)) +
  geom_tile(aes(fill = n), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f", n)), vjust = 1) +
  scale_fill_gradient(low = "ghostwhite", high = "#7F89B8", space = "Lab") +
  theme_bw() +
  theme(legend.position = "none") +
  scale_y_discrete(limits=rev) +
  xlab("Predicted ESS category") +
  ylab("Actual ESS category")
ggsave(filename = ".../Confusion matrix/ESS_training.pdf",
       plot = last_plot(),
       width = 2.4, height = 2.4, unit = "in", dpi = 300)

# ESS_TESTING
n <- c(8, 2, 1, 6)
df <- data.frame(label, pred, n)
ggplot(data = df, mapping = aes(x = label, y = pred)) +
  geom_tile(aes(fill = n), colour = "white") +
  geom_text(aes(label = sprintf("%1.0f", n)), vjust = 1) +
  scale_fill_gradient(low = "ghostwhite", high = "#7F89B8", space = "Lab") +
  theme_bw() +
  theme(legend.position = "none") +
  scale_y_discrete(limits=rev) +
  xlab("Predicted ESS category") +
  ylab("Actual ESS category")
ggsave(filename = ".../Confusion matrix/ESS_testing.pdf",
       plot = last_plot(),
       width = 2.4, height = 2.4, unit = "in", dpi = 300)
