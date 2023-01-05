
# Exploratory Data Analysis (EDA)

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 0A) Setting working directory

setwd('C:/Users/marcg/OneDrive/Escritorio/MTP-2023')

# 0B) Installing & loading packages from CRAN

# Package names

packages <- c('tidyverse',
              'magrittr',
              'DataExplorer',
              'funModeling',
              'GGally',
              'plotly',
              'nortest',
              'qqplotr')

# Install packages not yet installed

installed_packages <- packages %in% rownames(installed.packages())
if (any(installed_packages == FALSE)) {
  install.packages(packages[!installed_packages])
}

# Loading packages

invisible(lapply(X = packages, 
                 FUN = library, 
                 character.only = TRUE))

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 1) Loading the data

data <- read.table('data/data.csv', 
                   header = TRUE, 
                   sep = ',', 
                   row.names = NULL) %>%
  select(pH, temp, osm, qvcd)

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 2) Initial exploration

introduce(data)

plot_intro(data) +
  theme_bw() +
  theme(axis.title.y = element_text(vjust = +3),
        axis.title.x = element_text(vjust = -0.75))

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 3) Studying the distribution of the variables

# 3.1) Predictors (pH, temp, osm)

data_2 <- data
names <- c(1:3)
data_2[ , names] <- lapply(data_2[ , names] , factor)

plot_bar(data = data_2[ , 1:3],
         ggtheme = theme_bw(), 
         title = 'Distribution of the predictors') +
  theme(axis.title.y = element_text(vjust = +3),
        axis.title.x = element_text(vjust = -0.75))

# 3.2) Qvcd

# Histogram 

ggplot(data = data, mapping = aes(x = qvcd)) +
  geom_histogram(binwidth = 1, color = 'black', size = 0.5) +
  theme_bw() +
  ylim(c(0, 25)) +
  scale_x_continuous(breaks = round(seq(min(data$qvcd), max(data$qvcd), by = 5), 1)) +
  ggtitle(label = expression('Distribution of avg '*beta)) +
  labs(x = expression('avg '*beta*' [mM lactate/day]'), 
       y = 'count') +
  theme(axis.title.y = element_text(vjust = + 3),
        axis.title.x = element_text(vjust = -0.75))
  
# Density plot 

ggplot(data = data, mapping = aes(x = qvcd)) +
  geom_density(fill = 'firebrick', size = 0.5) +
  theme_bw() +
  scale_x_continuous(breaks = round(seq(min(data$qvcd), max(data$qvcd), by = 5), 1)) +
  ggtitle(label = expression('Distribution of avg '*beta)) +
  labs(x = expression('avg '*beta*' [mM lactate/day]'), 
       y = 'count') +
  theme(axis.title.y = element_text(vjust = + 3),
        axis.title.x = element_text(vjust = -0.75))

# Q-Q plot

ggplot(mapping = aes(sample = data$qvcd)) +
  stat_qq_point(size = 2) +
  stat_qq_line() +
  theme_bw() +
  ggtitle(label = expression('Q-Q plot of avg '*beta)) +
  labs(x = 'Theoretical quantiles',
       y = 'Sample quantiles') +
  theme(axis.title.y = element_text(vjust = + 3),
        axis.title.x = element_text(vjust = -0.75))

# Normality tests

# SW

shapiro.test(data$qvcd)

# AD

ad.test(data$qvcd)

# KS

ks.test(data$qvcd, "pnorm", mean = mean(data$qvcd), sd = sd(data$qvcd))

# 3) Studying the covariation between qvcd and each of our responses

# 3.1) pH vs qvcd

# All data

ggplot(data = data, mapping = aes(x = pH, y = qvcd)) +
  geom_point() +
  geom_smooth(method = 'lm', se = F) +
  theme_bw() +
  ggtitle(expression("pH vs"~beta)) +
  labs(y = expression(beta~"[mM lactate/day]"), 
       x = 'pH')
  
# Outliers removed

data <- data %>%
  filter(temp <= 35, osm == 0.3 & pH > 9) 

ggplot(data = data, mapping = aes(x = pH, y = qvcd)) +
  geom_point() +
  geom_smooth(se = F) +
  theme_bw() +
  ggtitle(expression("pH vs"~beta)) +
  labs(y = expression(beta~"[mM lactate/day]"), 
       x = 'pH')
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 4) Creating the binary categorical variable class + exploratory analysis

data_3 <- data %>%
  mutate(class = as.factor(ifelse(test = data$qvcd < 0.1, yes = 'safe', no = 'not safe'))) %>%
  select(pH, temp, osm, class)

prop.table(table(data_3$class))

# 4.1) pH vs temperature

ggplot(data = data_3, mapping = aes(x = pH, y = temp, color = class)) +
  geom_point(size = 3, position = position_jitter(h = 0.15, w = 0.15)) +
  theme_bw() +
  theme(axis.title.y = element_text(vjust = + 3),
        axis.title.x = element_text(vjust = -0.75))

# 4.2) pH vs osmolarity

ggplot(data = data_3, mapping = aes(x = pH, y = osm, color = class)) +
  geom_point(size = 3, position = position_jitter(h = 0.15, w = 0.15)) +
  theme_bw() +
  theme(axis.title.y = element_text(vjust = + 3),
        axis.title.x = element_text(vjust = -0.75))

# 4.3) osmolarity vs temperature

ggplot(data = data_3, mapping = aes(x = temp, y = osm, color = class)) +
  geom_point(size = 3, position = position_jitter(h = 0.15, w = 0.15)) +
  theme_bw() +
  theme(axis.title.y = element_text(vjust = + 3),
        axis.title.x = element_text(vjust = -0.75))

#) 4.4) pH vs temperature vs osmolarity


plot_ly(data_3, x = ~pH, y = ~temp, z = ~osm, color = ~class, colors = c('#BF382A', '#0C4B8E'))

