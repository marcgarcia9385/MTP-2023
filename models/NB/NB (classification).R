
# Naive Bayes (NB)

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 0A) Setting working directory

setwd('C:/Users/marcg/OneDrive/Escritorio/MTP-2023/')

# 0B) Installing & loading packages from CRAN

# Package names

packages <- c('tidyverse',
              'writexl',
              'vctrs',
              'rlang',
              'devtools', 
              'caTools', 
              'caret', 
              'e1071',
              'klaR',
              'naivebayes',
              'kernlab', 
              'ROSE', 
              'imbalance', 
              'themis', 
              'recipes',
              'plotly',
              'viridis')

# Install packages not yet installed

installed_packages <- packages %in% 
  rownames(installed.packages())

if (any(installed_packages == FALSE)) {
  install.packages(packages[!installed_packages])
}

# Loading packages

invisible(lapply(X = packages, 
                 FUN = library, 
                 character.only = TRUE))

# 0C) Installing and loading packages that are not available in CRAN

install_github('https://github.com/cran/DMwR', 
               force = T)

library(DMwR)

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 1) Data preparation 

# 1.1) Loading the data

data <- read.csv(file = 'data/data.csv', 
                 row.names = 1)

# 1.2) Creating the Class variable by fixing a threshold for the Qvcd

# NOTE: the positive class needs to go first

data$Class <- factor(x = ifelse(test = data$qvcd < 0.1, 
                                yes = 'safe', 
                                no = 'not.safe'),
                     levels = c('safe', 'not.safe'))

# 1.3) Removing the missing values and the Qvcd from the dataset

data <- data[ , -4]
data <- na.omit(data)
total_set <- data

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 2) Scaling/normalization

# total_set[ , -4] <- scale(total_set[ , -4])
# Scaling not needed for this algorithm

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 3) Modelling + evaluating performance

# 3.1) Random search with caret

# Check: http://topepo.github.io/caret/train-models-by-tag.html#supports-class-probabilities
# We will test Imbalanced and Balanced data, Laplace function, useKernel and adjust

# 3.1.1) Imbalanced training data, 5-fold CV, 10 repeats

# A) Accuracy, Kappa

set.seed(1234)

train_control_1a <- trainControl(method = 'repeatedcv', 
                                 number = 5, 
                                 repeats = 10,
                                 savePredictions = 'final')

(classifier_1a <- train(form= Class ~ ., 
                        data = total_set, 
                        trControl = train_control_1a, 
                        method = 'nb',
                        verbose = F))

# B) Sensitivity, Specificity, ROC

set.seed(1234)

train_control_1b <- trainControl(method = 'repeatedcv', 
                                 number = 5, 
                                 repeats = 10,
                                 savePredictions = 'final',
                                 classProbs = T, 
                                 summaryFunction = twoClassSummary)

(classifier_1b <- train(form = Class ~ ., 
                        data = total_set, 
                        trControl = train_control_1b, 
                        method = 'nb',
                        verbose = F))

# 3.1.2) Balanced training data, 5-fold CV, 10 repeats

# A) Accuracy, Kappa

set.seed(1234)

train_control_2a <- trainControl(method = 'repeatedcv', 
                                 number = 5, 
                                 repeats = 10, 
                                 savePredictions = 'final',
                                 sampling = 'up')

(classifier_2a <- train(form = Class ~ ., 
                        data = total_set, 
                        trControl = train_control_2a, 
                        method = 'nb',
                        verbose = F))

# B) Sensitivity, Specificity, ROC

set.seed(1234)

train_control_2b <- trainControl(method = 'repeatedcv', 
                                 number = 5, 
                                 repeats = 10, 
                                 savePredictions = 'final',
                                 sampling = 'up', 
                                 classProbs = T, 
                                 summaryFunction = twoClassSummary)

(classifier_2b <- train(form = Class ~ ., 
                        data = total_set, 
                        trControl = train_control_2b, 
                        method = 'nb', 
                        verbose = F))

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 3.2) Grid search with caret

# A) Defining the grid

grid <- as.data.frame(expand.grid(fL = seq(from = 0, 
                                              to = 1, 
                                              by = 0.1),
                                  usekernel = T,
                                  adjust = seq(from = 0,
                                                  to = 1, 
                                                  by = 0.1)))

# B) Accuracy, Kappa

set.seed(1234)

train_control_4a <- trainControl(method = 'repeatedcv', 
                                 number = 5, 
                                 repeats = 10,
                                 savePredictions = 'final',
                                 sampling = 'up')

(classifier_4a <- train(form = Class ~ ., 
                        data = total_set, 
                        trControl = train_control_4a, 
                        method = 'nb',
                        tuneGrid = grid,
                        verbose = F))

# C) Sensitivity, Specificity, ROC

set.seed(1234)

train_control_4b <- trainControl(method = 'repeatedcv', 
                                 number = 5, 
                                 repeats = 10, 
                                 savePredictions = 'final',
                                 sampling = 'up',
                                 classProbs = T, 
                                 summaryFunction = twoClassSummary)

(classifier_4b <- train(form = Class ~ ., 
                        data = total_set, 
                        trControl = train_control_4b, 
                        method = 'nb',
                        tuneGrid = grid,
                        verbose = F))

# D) Heatmaps

# D.1) Accuracy

(p1 <- ggplot(data = classifier_4a$results, 
              mapping = aes(x = fL, 
                            y = adjust, 
                            fill = Kappa)) +
    geom_tile() +
    theme_bw() +
    ylim(c(0.1, 1)) +
    scale_fill_viridis(discrete = F))

# D.2) Kappa

(p2 <- ggplot(data = classifier_4a$results, 
              mapping = aes(x = fL, 
                            y = adjust, 
                            fill = Accuracy)) +
    geom_tile() +
    theme_bw() +
    ylim(c(0.1, 1)) +
    scale_fill_viridis(discrete = F))

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 4) Final model

system.time(classifier_f <- NaiveBayes(formula = Class ~ ., 
                                       data = upSample(x = total_set[, -ncol(total_set)],
                                                       y = total_set$Class),
                                       fL = 0.5,
                                       usekernel = T,
                                       adjust = 0.7))

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 5) New predictions

# 5.1) Predictions A (screening) 

# Enter here the experimental conditions that you wish to predict:

prediction_data_1 <- tribble(
  
  ~cond,   ~pH,  ~temp,  ~osm,
  
      1,     9,     55,   0.3,
      2,  11.5,     30,   1.0,
      3,   9.5,     38,   1.0,
      4,  9.75,     47,   0.3,
      5,    11,     35,   0.4,
      6,  10.2,     46,   0.3,
      7,     9,     30,   0.3,
      8,     9,     45,   0.3,
      9,   9.5,     45,   0.3,
     10,    10,     45,   0.3,
     11,     9,     42,   0.6
  
)

prediction_data_1 <- prediction_data_1 %>% 
  slice(rep(1:n(), each = 100000))


system.time(predictions_1 <- predict(object = classifier_f,
                                     newdata = prediction_data_1,
                                     type = 'class'))

prediction_data_1$Class <- predictions_1$class

prediction_data_1b <- prediction_data_1 %>%
  group_by(cond) %>%
  count(Class) %>%
  mutate(prop = prop.table(n))

View(prediction_data_1b)

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 5.2) Predictions B (graphical representation)

# Define the sequences for the predictors that you want to test:

pH <- seq(from = 9, 
          to = 11.5, 
          length.out = 60)

temp <- seq(from = 30, 
            to = 55, 
            length.out = 60)

osm <- seq(from = 0.3, 
           to = 1.4, 
           length.out = 60)

# Crossing all the possible combinations

prediction_data_2 <- as.data.frame(crossing(pH, temp, osm))

# Making the predictions

predictions_2 <- predict(object = classifier_f,
                         newdata = prediction_data_2,
                         type = 'class')

prediction_data_2$Class <- predictions_2$class

prediction_data_2$Class.2 <- factor(x = prediction_data_2$Class, 
                                    levels = c('not.safe', 'safe'),
                                    labels = c(0, 1))

# Filtering the safe conditions

prediction_data_2_safe <- prediction_data_2 %>%
  filter(Class == 'safe')

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 5.3) Plotting predictions

# 5.3.1) 3D plots

# A) pH vs temp (when osm = 0.3 M NaCl)

(p3 <- ggplot() +
   geom_point(data = filter(prediction_data_2, osm <= 0.3), 
              mapping = aes(x = pH, 
                            y = temp, 
                            color = Class), 
              size = 2) +
   scale_color_manual(values = c('springgreen4', 'firebrick')) +
   theme_bw())

# B) pH vs osm (when temp is 30)

(p4 <- ggplot() +
    geom_point(data = filter(prediction_data_2, temp <= 30), 
               mapping = aes(x = pH, 
                             y = osm, 
                             color = Class),
               size = 2) +
    scale_color_manual(values = c('springgreen4', 'firebrick')) +
    theme_bw())
  
# C) osm vs temp (when pH is 9)

(p5 <- ggplot() +
    geom_point(data = filter(prediction_data_2, pH <= 9), 
               mapping = aes(x = osm, 
                             y = temp, 
                             color = Class),
               size = 2) +
      scale_color_manual(values = c('springgreen4', 'firebrick')) +
      theme_bw())
    
# 5.3.2) 4D plots

# A) 4D plot (all predictions)

(p6 <- plot_ly(data = prediction_data_2, 
               x = ~pH, 
               y = ~temp, 
               z = ~osm, 
               color = ~Class, 
               colors = c('springgreen4', 'firebrick')))

# B) 4D plot (safe predictions)

(p7 <- plot_ly(data = filter(prediction_data_2, Class == 'safe'), 
               x = ~pH, 
               y = ~temp, 
               z = ~osm, 
               color = ~Class, 
               colors = c('springgreen4')))

# 4D plot (not.safe predictions)

(p8 <- plot_ly(data = filter(prediction_data_2, Class == 'not.safe'), 
               x = ~pH, 
               y = ~temp, 
               z = ~osm, 
               color = ~Class, 
               colors = c('firebrick')))

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 5.4) Saving predictions

write.csv(x = prediction_data_2, 
          file = 'models/NB/prediction_data.csv')

write.csv(x = prediction_data_2_safe, 
          file = 'models/NB/prediction_data_safe.csv')
