
library(agricolae)

value <- c(0.81, 0.59, 0.9, 0.8, 0.85, 0.94, 0.92, 0.81, 0.96, 0.9, 0.95, 0.95, 0.84, 0.68, 0.94, 0.81, 0.95, 0.85, 0.91, 0.78, 0.91, 0.91, 0.90, 1.01)
parameter <- as.factor(rep(c('acc', 'kappa', 'ROC', 'sens', 'spec', 'sens_spec'), 4))
algorithm <- as.factor(rep(c('nb', 'svm', 'rf', 'ann'), each = 6))

data <- data.frame(value, parameter, algorithm)

ggplot(data, aes(fill = algorithm, y = value, x = parameter)) + 
  geom_bar(position = 'dodge', stat = 'identity') +
  theme_bw() +
  scale_fill_viridis(discrete = T) +
  geom_text(aes(label = value), vjust = 1.6, color = 'black',
                                              position = position_dodge(0.9), size = 2.5) +
  theme(axis.title.y = element_text(vjust = + 3),
        axis.title.x = element_text(vjust = -0.75))

 
# ANOVA tests

# accuracy





