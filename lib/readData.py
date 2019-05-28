import nltk
import torch
from lib.wordEmbed import Embedding

class ReadData:
	def __init__(self,dataDir,wordEmbed):
		self.device = torch.device("cuda:0")
		embedDim=300
		self.wordEmbedding = wordEmbed.getEmbed()
		self.wordIdxDic = wordEmbed.getWordIdxDic()

		fin = open(dataDir,'r')
		self.dataCount = 0
		self.dataIdx = 0
		#self.senTensorList = []
		#self.answerList = []
		self.dataPairList = []

		print("Read query data")

		while(True):
			line = fin.readline().rstrip()
			if(line == ''):
				break
			if not line:
				break
			tokens = line.split('\t')

			tokenizedSen = nltk.word_tokenize(tokens[1])
			tokenizedSen = [x.lower() for x in tokenizedSen]
			for i in range(len(tokenizedSen)):
				if(tokenizedSen[i] not in self.wordIdxDic.keys()):
					tokenizedSen[i] = '<unk>'
			idxs = [self.wordIdxDic[w] for w in tokenizedSen]
			idxTensor = torch.LongTensor(idxs)
			senTensor = self.wordEmbedding(idxTensor).to(self.device)
			#self.senTensorList.append(senTensor)

			if(tokens[2]=="T"):
				self.dataPairList.append((senTensor,1,senTensor.size(0)))
				#self.answerList.append(1)
			else:
				self.dataPairList.append((senTensor,0,senTensor.size(0)))
				#self.answerList.append(0)

			self.dataCount += 1
		print("reading done")
		# sort by decreasing length
		self.dataPairList = sorted(self.dataPairList,key=lambda x:x[2],reverse=True)
		#print(self.dataPairList)	
	def getBatchTensor(self,batchSize):
		batchPairList = self.getBatchData(batchSize)

		tensorList = []
		answerList = []
		lenList = []

		for i in range(len(batchPairList)):
			tensorList.append(batchPairList[i][0])
			answerList.append(batchPairList[i][1])
			lenList.append(batchPairList[i][2])
		dataTensor = torch.nn.utils.rnn.pad_sequence(tensorList)
		dataTensor = dataTensor.transpose(0,1).to(self.device)

		resultTensor = torch.Tensor(answerList).to(self.device)

		return dataTensor,resultTensor,lenList


	def getBatchData(self,batchSize):
		if(self.dataIdx+batchSize > self.dataCount-1):
			self.dataIdx = 0
			return self.dataPairList[-batchSize:]
		else:
			self.dataIdx += batchSize
			return self.dataPairList[self.dataIdx-batchSize:self.dataIdx]
	def getDataCount(self):
		return self.dataCount
