# Fasta2svm
***********************
&emsp;&emsp;将fasta序列kmer分词，然后通过word2vec训练词向量。再将序列中每一个词取出对应的向量，对一条序列的所有的词向量取平均为该序列的向量。 
然后将生成的csv文件给svm进行优化分类。
### 输出文件
* 词向量模型
* svm模型
* 词向量模型可以通过 -premodel 导入自定义的词向量模型。
*****************************
### train 参数设置

|参数|取值|
|:-|:-|  
|-t|xxx.fasta|    
|-o|model|   
-kmer/-k|       	 kmer分词的k取值. default=3  
-b     |           当模型中不存在该词的词向量的时候用b填充，default=0  
-sg     |     		0:cbow   1:skip-gram .default =1 
-hs      |     	0:Negative Sampling   1:Hierarchical softmax .default=0  
-window_size|   	窗口大小  
-model_name      |	词向量模型文件名  default=model.model  
-hidden_size|    	词向量的维度， default=100  
-mms   |		   是否使用归一化. default=False    
-ss     | 		 是否使用标准化. default=False 
-cv   |		交叉验证折数, default10-fold 
-n_job   |		线程数 default=(-1)最大线程  
-splite/-s| 0:kmer splite; 1:normal splite .default=0
-premodel|加载预训练词向量模型
-iter|word2vec迭代次数
-grad|是否网格搜索；默认False
*********************************
### predict 参数设置

|参数|取值|
|:-|:-|  
|-t|类别1.fasta;类别2.fasta 。。。|    
-m|      模型
-kmer/-k|       	 kmer分词的k取值. default=3  
-b     |           当模型中不存在该词的词向量的时候用b填充，default=0  
-em      |	词向量模型文件名  default=model.model  
-mms   |		   是否使用归一化. default=False    
-ss     | 		 是否使用标准化. default=False 
-splite/-s| 0:kmer splite; 1:normal splite .default=0

*********************************

* 常用的调节参数  
-grad： 网格搜索  
-premodel：加载预训练词向量模型  
-splite：分词的方式，默认是kmer分词  
-cv：交叉验证的折数  
-mms：归一化
*************************
### Example
train：
```py
python train.py -i 1.fasta 2.fasta -o svm.pkl
```
test：
```py
python predict.py -t 1.fasta 2.fasta -m svm.pkl -em model.model
```

