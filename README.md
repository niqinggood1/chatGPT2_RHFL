# gpt2_chat

## 使用方法

### 预处理数据
先要将数据转为一行一位发言者的数据形式  

  运行 raw_corpur_process.py 将保险的问答数据转化为对话对  
  运行preprocess.py，对data/train.txt对话语料进行tokenize，然后进行序列化保存到data/train.pkl。train.pkl中序列化的对象的类型为List[List],记录对话列表中,每个对话包含的token。  
  python preprocess.py --train_path data/train.txt --save_path data/train.pkl


### 训练数据编码
python gpt2_data_preprocess.py --vocab_path gpt2-chinese-cluecorpussmall --train_path data/big_common.txt --save_path data/cluecorpus_common.pkl


### 训练模型
运行train.py,使用预处理后的数据，对模型进行自回归训练，模型保存在根目录下的model文件夹中
python train.py --epochs 40 --batch_size 8 --device 0,1 --train_path data/train.pkl

### 训练
python gpt2_big_train.py --pretrained_model smallmodel/min_ppl_model  --vocab_path gpt2-chinese-cluecorpussmall --save_model_path smallmodel --epochs 10 --batch_size 60 --device 0,1 --train_path data/cluecorpus_common.pkl --lr 0.000052 --warmup_steps 400


python3 gpt2_data_preprocess.py --vocab_path Wenzhong2.0-GPT2-3.5B-chinese --train_path data/vhs_medical.txt --save_path data/vhs_medical.pkl

###  模型交互
执行如下命令，进行对话
python interact.py --no_cuda --model_path model/epoch1001  (使用cpu生成，速度相对较慢)
或
python interact.py --model_path model/epoch1001  --device 0 (指定0号GPU进行生成，速度相对较快)





