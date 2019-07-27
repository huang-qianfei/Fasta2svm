# Fasta2svm

python fasta2svm.py  
* 参数设置

|参数|取值|
|:-:|:-:|  
|-trainfasta|xxx.fasta|    
|-trainword|训练集分词的文件名，默认trainword.txt|   
-trainpos|    	正例数  
-trainneg|       	反例数  
-traincsv|       	 默认 train.csv  
-kmer/-k|       	 kmer分词的k取值, default=3  
-b     |               	当模型中不存在该词的词向量的时候用b填充，默认0  
-sg     |     		0 -cbow   1 skip-gram（default =1）  
-hs      |     	0 -负采样   1 Hierarchical softmax （default=0）  
-window_size|   	窗口大小  
-model      |	词向量文件名  默认model.model  
-hidden_size|    	词向量的维度， default=100  
-testfasta  | 	 xxx.fasta
-testword   |	测试集分词文件名字，默认testword.txt  
-testpos   | 	正例数  
-testneg  |		反例数  
-testcsv | 		默认 test.csv  
-mms   |		默认False  是否使用归一化  
-ss     | 		默认False  是否使用标准化  
-cv   |		交叉验证折数  
-n_job   |		线程数 默认使用最大线程  
*********************************
* 必须设置的参数有三个  
-trainfasta  
-trainpos  
-trainneg  
*************************
### example  
* python fatsa2svm.py -trainfasta   -trainpos num  -trainneg num
* python fatsa2svm.py -trainfasta   -trainpos num  -trainneg num  -testfasta   -testpos num  -testneg num -mms True

