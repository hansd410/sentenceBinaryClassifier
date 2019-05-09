import sys

fin = open(sys.argv[1],'r')

answerList = []
resultList = []
while True:
	line = fin.readline()
	if not line:
		break
	tokens = (line.rstrip()).split('\t')
	answerList.append(float(tokens[0]))
	resultList.append(float(tokens[1]))

fout = open("threshold.txt",'w')
for j in range(10):
	resultStr = ""
	thres = (j+1)/10
	resultStr += "thres\t"+str(thres)+"\t"

	tt=0
	tf=0
	ft=0
	ff=0

	for i in range(len(answerList)):
		if(resultList[i]>=thres):
			if(answerList[i]==1):
				tt+=1
			else:
				tf+=1
		else:
			if(answerList[i]==1):
				ft+=1
			else:
				ff+=1
	if(tt+tf==0):
		precision=0
	else:
		precision=tt/(tt+tf)
	if(tt+ft==0):
		recall=0
	else:
		recall=tt/(tt+ft)
	if(ff+ft==0):
		fprecision=0
	else:
		fprecision=ff/(ff+ft)
	if(ff+tf==0):
		frecall=0
	else:
		frecall=ff/(ff+tf)
	resultStr += "tt/tf/ft/ff\t"+str(tt)+"\t"+str(tf)+"\t"+str(ft)+"\t"+str(ff)+"\tpre/rec/fpre/frec\t"+str(precision)+"\t"+str(recall)+"\t"+str(fprecision)+"\t"+str(frecall)+"\n"
	fout.write(resultStr)

#	print("tt:"+str(tt))
#	print("tf:"+str(tf))
#	print("ft:"+str(ft))
#	print("ff:"+str(ff))
#	print("precision:"+str(float(tt/(tt+tf))))
#	print("recall:"+str(float(tt/(tt+ft))))
#	print("false precision:"+str(float(ff/(ff+ft))))
#	print("false recall:"+str(float(ff/(ff+tf))))
