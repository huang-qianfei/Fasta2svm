# Fasta2svm
python fasta2svm.py   
-trainfasta     	训练集fasta文件  
-trainword    	训练集分词的文件名，默认trainword.txt  
-trainpos    	正例数  
-trainneg       	反例数  
-traincsv       	 csv名字 默认train.csv  
-kmer/-k       	 kmer分词的k取值, default=3  
-b                    	当模型中不存在该词的词向量的时候用b填充，默认0  
-sg          		0 -cbow   1 skip-gram（default =1）  
-hs           	0 -负采样   1 层序softmax ，default=0  
-window_size   	上下文最大距离  
-model      	词向量模型名字  默认model.model  
-hidden_size    	词向量的生成维度， default=100  
-testfasta   	测试集fasta文件  
-testword   	测试集分词文件名字，默认testword.txt  
-testpos    	测试集正例  
-testneg  		反例  
-testcsv  		测试集 csv 默认 test.csv  
-mms   		默认False  是否使用归一化  
-ss      		默认False  是否使用标准化  
-cv   		交叉验证折数  
-n_job   		线程数 默认使用最大线程  
*********************************
必须设置的参数有三个  
-trainfasta  
-trainpos  
-trainneg  
