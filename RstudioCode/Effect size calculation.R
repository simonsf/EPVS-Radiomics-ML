rm(list = ls())

# 加载必要的包
library(effsize)

# 读取数据
data <- read.csv('C:/Windows/Temp/AA/MoCA_6features.csv')

# 获取特征列名
features <- colnames(data)[-1]  # 排除第一列Label

# 为每个特征计算效应量
results <- lapply(features, function(feature) {
  # 提取两组的特征数据
  group1 <- data[data$Label == 1, feature]
  group2 <- data[data$Label == 0, feature]
  
  # 计算效应量
  cd <- cohen.d(group1, group2)
  
  # 返回结果
  return(data.frame(
    Feature = feature,
    EffectSize = cd$estimate,
    Magnitude = cd$magnitude,
    CI_lower = cd$conf.int[1],
    CI_upper = cd$conf.int[2]
  ))
})

# 合并所有结果
results_df <- do.call(rbind, results)
write.csv(results_df, file = "C:/Windows/Temp/AA/effect_size_MoCA.csv", row.names = FALSE)

# 查看结果
print(results_df)
