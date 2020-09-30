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

|parameters|values|
|:-|:-|  
|-t|class1.fasta;class2.fasta 。。。|    
|-o|model name|   
-kmer/-k|       	 k-mer,default:k=3  
-b     |           padding,default:b=0  
-sg     |     		0:cbow ,  1:skip-gram .default =1 
-hs      |     	0:Negative Sampling,   1:Hierarchical softmax .default=0  
-window_size|   	word2vec window size  
-model_name      |	embedding model name , default=model.model  
-hidden_size|    	dim of embedding vector, default=100  
-mms   |		   MinMaxScaler, default=False    
-ss     | 		 StandardScaler, default=False 
-cv   |		x
-n_job   |		 default=(-1)
-splite/-s| 0:kmer splite, 1:normal splite .default=0
-premodel|pre-train model
-iter|iter-num of word2vec
-grad|GridSearch, default:False
*********************************
### predict 参数设置

|parameters|values|
|:-|:-|  
|-t|class1.fasta;class2.fasta 。。。|    
-m|      classifier model
-kmer/-k|       	 k-mer, default:k=3  
-b     |           padding,default:b=0  
-em      |	embedding model,  default=model.model  
-mms   |		   MinMaxScaler, default=False    
-ss     | 		 StandardScaler, default=False 
-splite/-s| 0:kmer splite, 1:normal splite .default=0

*********************************

* 常用的调节参数  
-grad： 网格搜索  
-premodel：加载预训练词向量模型  
-splite：分词的方式，默认是kmer分词  
-cv：交叉验证的折数  
-mms：归一化
*************************
### Example
* 这里1.fasta ，2.fasta分别对应这一个类别的fasta文件； 
train：
```py
python train.py -i 1.fasta 2.fasta -o svm.pkl
```
predict：
```py
python predict.py -t 1.fasta 2.fasta -m svm.pkl -em model.model
```

