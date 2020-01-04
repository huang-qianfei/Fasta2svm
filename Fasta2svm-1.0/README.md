# Fasta2svm
***********************
&emsp;&emsp;将fasta序列kmer分词，然后通过word2vec训练词向量。再将序列中每一个词取出对应的向量，对一条序列的所有的词向量取平均为该序列的向量。 
然后将生成的csv文件给svm进行优化分类。
### 输出文件
* 分词文件
* 词向量模型
* csv文件
* 可以将生成的csv文件单独拿出来测试其他的分类器。词向量模型可以通过 -spmodel 导入自定义的词向量模型。
*****************************
### 参数设置

|参数|取值|
|:-|:-|  
|-trainfasta|训练集：xxx.fasta|    
|-trainword|训练集分词的文件名. default=trainword.txt|   
-trainpos|    	 训练集正例数  
-trainneg|       	训练集反例数  
-traincsv|       	 训练集的csv文件名：default= train.csv  
-kmer/-k|       	 kmer分词的k取值. default=3  
-b     |           当模型中不存在该词的词向量的时候用b填充，default=0  
-sg     |     		0:cbow   1:skip-gram .default =1 
-hs      |     	0:Negative Sampling   1:Hierarchical softmax .default=0  
-window_size|   	窗口大小  
-model      |	词向量模型文件名  default=model.model  
-hidden_size|    	词向量的维度， default=100  
-testfasta  | 	  测试集：xxx.fasta
-testword   |	测试集分词文件名，default=testword.txt  
-testpos   | 	 测试集正例数  
-testneg  |		测试集反例数  
-testcsv | 		 测试集的csv文件名：default=test.csv  
-mms   |		   是否使用归一化. default=False    
-ss     | 		 是否使用标准化. default=False 
-cv   |		交叉验证折数, default10-fold 
-n_job   |		线程数 default=(-1)最大线程  
-splite/-s| 0:kmer splite; 1:normal splite .default=0
-spmodel|加载预训练词向量模型
-iter|word2vec迭代次数
-grad|是否网格搜索；默认False
*********************************
* 必须设置的参数有三个  
-trainfasta  
-trainpos  
-trainneg  
* 常用的调节参数  
-grad： 网格搜索  
-spmodel：加载预训练词向量模型  
-splite：分词的方式，默认是kmer分词  
-cv：交叉验证的折数  
-mms：归一化
*************************
### Example
交叉验证用法(默认10-fold)：
```py
python fasta2svm.py -trainfasta xxx.fasta  -trainpos num  -trainneg num
```
独立测试用法：
```py
python fasta2svm.py -trainfasta xxx.fasta  -trainpos num  -trainneg num  -testfasta XXX.fasta  -testpos num  -testneg num -mms True
```

