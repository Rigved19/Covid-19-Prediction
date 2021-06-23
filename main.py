model=MLP()    #model is an object of class MLP

#Training the model
NeuralNet1 = Layer(2,[2,5],activation=['linear'])
NeuralNet2 = Layer(4,[12,16,16,2],activation=['relu','relu','linear'])
epochs=1500
lr1=0.1
lr2=0.005
model.Train(Xnet1_train_scaled.T, Xnet2_train_scaled.T, Ytrain_scaled.T, NeuralNet1, NeuralNet2, epochs, lr1, lr2, printcost=True)
trainedNet1,trainedNet2 = loadmodel.load_weights()

#Cheecking on Validation Set
Ypred = model.Predict(Xnet1_train_scaled.T, Xnet2_train_scaled.T, trainedNet1, trainedNet2)
print('TRAINING LOSS :'+str((mse(Ypred.T,Ytrain_scaled))))

#Predicting
#Using model weights for test cases
test_days=test_days.reshape(1,test_days.shape[0])
Ypred_test = model.Test(Xnet1_test_scaled.T, Xnet2_test_scaled.T, test_days, trainedNet1, trainedNet2)
#Loss for Test Cases
print('TEST LOSS : ' + str(mse(Ypred_test,Ytest_scaled.T)))

