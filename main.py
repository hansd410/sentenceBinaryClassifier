from lib.readData import ReadData
from lib.model import Net
from lib.wordEmbed import Embedding
import argparse
import torch
import torch.optim as optim

def get_args():
	parser = argparse.ArgumentParser(description="aligner parameter")
	parser.add_argument('--model',default="0", help="modelName")
	parser.add_argument('--mode',default="train", help="train or test")
	parser.add_argument('--debug',default="False", help="debug mode")
	parser.add_argument('--gpu',default="0", help="gpu id")
	parser.add_argument('--oneHotDim',default='100000', help="oneHot dim")
	parser.add_argument('--embedDim',default='300', help="embedding dim")
	parser.add_argument('--hiddenDim',default='300', help="g hidden dim")
	parser.add_argument('--trainDir',default='data/query/train_0',help="train data dir")
	parser.add_argument('--testDir',default='data/query/test_0',help="test data dir")

	return parser.parse_args()

args=get_args()

epochNum = 10
batchSize = 50
learningRate = 0.001
embedDim = 300

print("network initialization")
net=Net().cuda()
print("init done")

print("read embedding")
wordEmbed=Embedding(embedDim)
print("embedding done")

print("reading data")
trainData = ReadData(args.trainDir,wordEmbed)
testData = ReadData(args.testDir,wordEmbed)
print("data reading done")

trainDataCount = trainData.getDataCount()
testDataCount = testData.getDataCount()
trainIteration = epochNum*trainDataCount//batchSize + 1
trainEpochIteration = trainDataCount//batchSize + 1
testEpochIteration = testDataCount//batchSize + 1

if(args.mode =="train"):
	fout = open("learningCurve.txt",'w')
	for i in range(trainIteration):
		data=trainData.getBatchTensor(batchSize)
		_,answerTensor,_ = data
		loss = torch.sum((answerTensor-net(data))**2)
		optimizer = optim.Adam(net.parameters(),lr=learningRate)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if(i==0 or i%99==0):
			print("test begins at "+str(i)+"th iteration")
			testLoss = 0
			for j in range(testEpochIteration):
				data = testData.getBatchTensor(batchSize)
				_,answerTensor,_ = data
				testLoss += torch.sum((answerTensor-net(data))**2)
			testLoss = testLoss/testDataCount
			print("testLoss\t"+str(testLoss))
			trainLoss = 0
			for j in range(trainEpochIteration):
				data = trainData.getBatchTensor(batchSize)
				_,answerTensor,_ = data
				trainLoss += torch.sum((answerTensor-net(data))**2)
			trainLoss = trainLoss/trainDataCount
			print("trainLoss\t"+str(trainLoss))
			fout.write("testLoss\t"+str(testLoss.item())+"\ttrainLoss\t"+str(trainLoss.item())+"\n")
			torch.save(net.state_dict(),'/mnt/data/hansd410/exo/'+str(i))
else:
	print("test begins")
	net.load_state_dict(torch.load('/mnt/data/hansd410/exo/'+args.model))
	answerList = []
	resultList = []
	for j in range(testEpochIteration):
		data = testData.getBatchTensor(batchSize)
		_,answerTensor,_ = data
		resultTensor=net(data)
		for k in range(answerTensor.size(0)):
			answerList.append(answerTensor[k].item())
			resultList.append(resultTensor[k].item())
#			if(resultTensor[k]<0.5):
#				resultList.append(0)
#			else:
#				resultList.append(1)
	
	fout = open("result.txt",'w')
	for i in range(len(answerList)):
		fout.write(str(answerList[i])+"\t"+str(resultList[i])+"\n")
