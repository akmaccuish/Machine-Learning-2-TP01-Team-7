# We have seen that we can fit an SVM with a non-linear kernel in order
# to perform classification using a non-linear decision boundary. We will
# now see that we can also obtain a non-linear decision boundary by
# performing logistic regression using non-linear transformations of the
# features.
rm(list=ls())
# (a) Generate a data set with n = 500 and p = 2, such that the observations 
#     belong to two classes with a quadratic decision boundary
#     between them. For instance, you can do this as follows:
x1 <- runif (500) - 0.5
x2 <- runif (500) - 0.5
y <- 1 * (x1^2 - x2^2 > 0)
mydf <- data.frame(x1=x1,x2=x2,y=as.factor(y))
train

#(b) Plot the observations, colored according to their class labels.
#    Your plot should display X1 on the x-axis, and X2 on the yaxis.
plot(x1,x2, col = ifelse(y == 1, 'red', 'blue'))
legend('topright',c('1','0'), col=c('red','blue'), pch=20)

#(c) Fit a logistic regression model to the data, using X1 and X2 as
#    predictors.

logisticreg <- glm(y~x1+x2, family=binomial, data=mydf)
summary(logisticreg)

# (d) Apply this model to the training data in order to obtain a predicted class label for each training observation. Plot the observations, colored according to the predicted class labels. The
#     decision boundary should be linear.
n <- nrow(mydf)
trainIndex <- sample(1:n, size = n*0.8)
train <- mydf[trainIndex,]
test <- mydf[-trainIndex,]
log.pred <- predict(logisticreg, newdata = train)
plot(train$x1,train$x2, col = ifelse(log.pred >= 0, 'red', 'blue'))
legend('bottomright',c('1','0'), col=c('red','blue'), pch=20)

# (e) Now fit a logistic regression model to the data using non-linear
#     functions of X1 and X2 as predictors (e.g. X1^2, X1*X2, log(X2), and so forth).
log1 <- glm(y~(x1*x2), family=binomial, data=mydf)

# (f) Apply this model to the training data in order to obtain a predicted class label for each training observation. Plot the observations, colored according to the predicted class labels. The
# decision boundary should be obviously non-linear. If it is not,
# then repeat (a)-(e) until you come up with an example in which
# the predicted class labels are obviously non-linear.
log.pred <- predict(log1, newdata = train)
plot(train$x1,train$x2, col = ifelse(log.pred >= 0, 'red', 'blue'))
legend('bottomright',c('1','0'), col=c('red','blue'), pch=20)

# (g) Fit a support vector classifier to the data with X1 and X2 as
# predictors. Obtain a class prediction for each training observation. Plot the observations, colored according to the predicted
# class labels.


# (h) Fit a SVM using a non-linear kernel to the data. Obtain a class
# prediction for each training observation. Plot the observations,
# colored according to the predicted class labels.


# (i) Comment on your results