import jieba
import pandas as pd
#import gensim.models

#读取数据
df_entertainment = pd.read_csv(r'E:\Data_sets\text_ana\entertainment_news.csv',encoding='utf-8')
df_entertainment = df_entertainment.dropna()

df_finance = pd.read_csv(r'E:\Data_sets\text_ana\finance_news.csv',encoding='utf-8')
df_finance = df_finance.dropna()

df_international = pd.read_csv(r'E:\Data_sets\text_ana\international_news.csv',encoding='utf-8')
df_international = df_international.dropna()

df_technology = pd.read_csv(r'E:\Data_sets\text_ana\technology_news.csv',encoding='utf-8')
df_technology = df_technology.dropna()


entertainment = df_entertainment.content.values[:20000]
finance = df_finance.content.values[:20000]
international = df_international.content.values[:20000]
technology = df_technology.content.values[:20000]

#获取停用词
stopwords = pd.read_csv('E:/Data_sets/text_ana/stopwords.txt',
                        index_col=False,
                        quoting=3,sep="\t",
                        names=['stopword'], 
                        encoding='utf-8')

stopwords=stopwords['stopword'].values

# 构建数据集

def preprocess(content, sentences, category):

    for line in content:
        try:
            segs = jieba.lcut(line)   
            segs = filter(lambda x:len(x)>1, segs)
            segs = filter(lambda x:x not in stopwords, segs)
            sentences.append((' '.join(segs),category))
        except Exception as e:
            print('Error:'+ str(e))
            continue
    
sentences = []
preprocess(entertainment,sentences,'entertainment \n')
preprocess(finance,sentences,'finance \n')
preprocess(international,sentences,'international \n')
preprocess(technology,sentences,'technology \n') 

#print(sentences[:20])

#存储分词后的文本
print ("writing data to fasttext format...")
out = open(r'E:\Data_sets\text_ana\jieba_train_data_utf.txt', 'w')
for sentence in sentences:
    out.writelines(sentence)
print ("done!")


from gensim.models import word2vec
import time


start = time.time()
sentences = word2vec.Text8Corpus(r'E:\Data_sets\text_ana\jieba_train_data_utf.txt')
model = word2vec.Word2Vec(sentences, size=200)
y1 = model.similarity("演员", "艺人")
print ("[演员]和[艺人]的相似度为：", y1)
print( "--------")

y2 = model.most_similar(u"文化", topn=10)  # 20个最相关的
print ("和[文化]最相关的词有：\n")
for item in y2:
    print (item[0], item[1])
print( "--------")
end = time.time()
longtime = end - start
print('time:'+str(longtime))


