# installing neuralnet package
# install.packages('neuralnet')

require(neuralnet)

hours = c(20, 10, 30, 30, 40, 70)
mocktest = c(90, 20, 10, 10, 80, 60)
passed = c(1, 0, 0, 0, 1, 1)

df = data.frame(hours, mocktest, passed)

# neural network of classification
nn = neuralnet(passed~hours+mocktest, data=df, hidden = c(3,2), act.fct = 'logistic', linear.output = FALSE)
plot(nn)

# test data and dataframe
testHours = c(20, 20, 30)
testMock = c(80, 90, 10)
test = data.frame(testHours, testMock)

# predicting 
predict = compute(nn, test)
predict$net.result
prob <- predict$net.result
pred <- ifelse(prob>0.5, 1, 0)
pred
