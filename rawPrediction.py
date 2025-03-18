import numpy as np
import pandas as pd
#设置 pandas 在显示数据时，不限制显示的列数，即显示所有列。
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
#设置 matplotlib 的绘图风格为 ggplot，这是一种类似于 R 语言中 ggplot2 包的绘图风格。
plt.style.use('ggplot')
import datetime
#LightGBM 是一个快速、高效的梯度提升框架
import lightgbm as lgb
#stats 模块，用于进行统计分析
from scipy import stats
#hstack 用于水平堆叠稀疏矩阵，csr_matrix 是一种压缩稀疏行矩阵格式
from scipy.sparse import hstack, csr_matrix
#train_test_split 用于将数据集划分为训练集和测试集，KFold 用于进行 K 折交叉验证。
from sklearn.model_selection import train_test_split, KFold
#生成词云图
from wordcloud import WordCloud
#用于统计元素的出现次数。
from collections import Counter
#获取停用词列表。
from nltk.corpus import stopwords
#用于生成 n-gram 序列。
from nltk.util import ngrams
#fidfVectorizer 用于将文本数据转换为 TF-IDF 特征矩阵，CountVectorizer 用于将文本数据转换为词频矩阵。
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
#一个用于自然语言处理的工具包
import nltk
nltk.download('stopwords')
stop = set(stopwords.words('english'))
from lightgbm import early_stopping
#用于与操作系统进行交互，例如文件和目录操作
import os
#plotly 是一个交互式可视化库，支持多种图表类型。
import plotly.offline as py
#允许在 Notebook 中直接显示交互式图表。
py.init_notebook_mode(connected=True)
#于创建各种图表对象。
import plotly.graph_objs as go
import plotly.tools as tls
#XGBoost 是另一个流行的梯度提升框架，常用于机器学习中的分类和回归任务。
import xgboost as xgb
#可能是代码编写时的重复操作
import lightgbm as lgb
#用于模型选择和评估。
from sklearn import model_selection
#用于计算分类模型的准确率。
from sklearn.metrics import accuracy_score
import json
#用于将字符串形式的 Python 表达式解析为抽象语法树
import ast
#用于对机器学习模型进行特征重要性分析和模型解释。
import eli5
#用于计算 SHAP 值，对模型的预测结果进行解释
import shap
#CatBoost 是一个基于梯度提升的机器学习库，对类别特征有很好的支持。
from catboost import CatBoostRegressor
#从 urllib.request 模块中导入 urlopen 函数，用于打开 URL 并读取数据。
from urllib.request import urlopen
#从 Pillow 库中导入 Image 类，用于图像处理
from PIL import Image
#用于对类别特征进行编码
from sklearn.preprocessing import LabelEncoder
import time
#用于计算回归模型的均方误差
from sklearn.metrics import mean_squared_error
#linear_model 模块中导入 LinearRegression 类，用于进行线性回归分析。
from sklearn.linear_model import LinearRegression
#可能是为了后续使用该模块中的其他线性模型。
from sklearn import linear_model
from joblib import dump
#——————————————————————————————————————————————————————————————————————————————————————————————
#数据清洗部分

train = pd.read_csv('MoviesData/rawPredictData/train.csv')
test = pd.read_csv('MoviesData/rawPredictData/test.csv')

# from this kernel: https://www.kaggle.com/gravix/gradient-in-a-box
#包含多个列名的列表
dict_columns = ['belongs_to_collection', 'genres', 'production_companies',
                'production_countries', 'spoken_languages', 'Keywords', 'cast', 'crew']
#该函数接受一个 DataFrame 对象 df 作为输入，将 dict_columns 列表中指定的列的数据从字符串转换为 Python 字典对象。
def text_to_dict(df):
    for column in dict_columns:
        #用于检查元素 x 是否为缺失值（NaN 或 None）。如果是缺失值，则返回一个空字典 {}。
        #ast.literal_eval(x) 函数将字符串 x 转换为 Python 字典对象。
        df[column] = df[column].apply(lambda x: {} if pd.isna(x) else ast.literal_eval(x) )
    return df
        
train = text_to_dict(train)
test = text_to_dict(test)

list_of_genres = list(train['genres'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
list_of_companies = list(train['production_companies'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
list_of_countries = list(train['production_countries'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
list_of_languages = list(train['spoken_languages'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
list_of_keywords = list(train['Keywords'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
list_of_cast_names = list(train['cast'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
list_of_cast_genders = list(train['cast'].apply(lambda x: [i['gender'] for i in x] if x != {} else []).values)
list_of_cast_characters = list(train['cast'].apply(lambda x: [i['character'] for i in x] if x != {} else []).values)
list_of_crew_names = list(train['crew'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
list_of_crew_jobs = list(train['crew'].apply(lambda x: [i['job'] for i in x] if x != {} else []).values)
list_of_crew_genders = list(train['crew'].apply(lambda x: [i['gender'] for i in x] if x != {} else []).values)
list_of_crew_departments = list(train['crew'].apply(lambda x: [i['department'] for i in x] if x != {} else []).values)
list_of_crew_names = train['crew'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values
#————————————————————————————————————————————————————————————————————————————————————————————
#改造belongs_to_collection这一列，将其拆分为collection_name和has_collection
train['collection_name'] = train['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != {} else 0)
train['has_collection'] = train['belongs_to_collection'].apply(lambda x: len(x) if x != {} else 0)

test['collection_name'] = test['belongs_to_collection'].apply(lambda x: x[0]['name'] if x != {} else 0)
test['has_collection'] = test['belongs_to_collection'].apply(lambda x: len(x) if x != {} else 0)

train = train.drop(['belongs_to_collection'], axis=1)
test = test.drop(['belongs_to_collection'], axis=1)
#分为了num_genres（题材数量）和all_genres（每部电影的题材列表）
#创建了15个新的属性，代表该电影是否为该题材genres_name 0/1最后消除genres这个属性
#将genres这一列分为num_genres（题材数量）和all_genres（每部电影的题材列表）
train['num_genres'] = train['genres'].apply(lambda x: len(x) if x != {} else 0)
train['all_genres'] = train['genres'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
#将前15个出现次数最多的题材放到top_genres这个列表里
top_genres = [m[0] for m in Counter([i for j in list_of_genres for i in j]).most_common(15)]

#创建15个新属性，每一列都是0或1，即代表该电影是否属于该题材
for g in top_genres:
    train['genre_' + g] = train['all_genres'].apply(lambda x: 1 if g in x else 0)

test['num_genres'] = test['genres'].apply(lambda x: len(x) if x != {} else 0)
test['all_genres'] = test['genres'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
for g in top_genres:
    test['genre_' + g] = test['all_genres'].apply(lambda x: 1 if g in x else 0)

#消除genres这个属性
train = train.drop(['genres'], axis=1)
test = test.drop(['genres'], axis=1)
#分为了num_companies（每部电影的赞助公司有几个）和all_production_companies（每部电影的所有制作公司）
#创建了30个新的属性，代表该电影是否为该制作公司所制作production_company_name 0/1
#最后消除production_companies,all_production_companies属性
#将production_companies分为num_companies（每部电影的赞助公司有几个）和all_production_companies
train['num_companies'] = train['production_companies'].apply(lambda x: len(x) if x != {} else 0)
train['all_production_companies'] = train['production_companies'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
#将出现次数最多的赞助公司名放入top_companies列表
#list_of_companies  [[],[],[]]
top_companies = [m[0] for m in Counter([i for j in list_of_companies for i in j]).most_common(30)]
#设置新的属性，每一列都是0或1，即代表该电影是否受该公司的赞助
for g in top_companies:
    train['production_company_' + g] = train['all_production_companies'].apply(lambda x: 1 if g in x else 0)
    
test['num_companies'] = test['production_companies'].apply(lambda x: len(x) if x != {} else 0)
test['all_production_companies'] = test['production_companies'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
for g in top_companies:
    test['production_company_' + g] = test['all_production_companies'].apply(lambda x: 1 if g in x else 0)

train = train.drop(['production_companies', 'all_production_companies'], axis=1)
test = test.drop(['production_companies', 'all_production_companies'], axis=1)
#分为了num_countries（国家数量）和all_countries（每部电影的国家数量）两个属性
#创建了25个新的属性，代表该电影是否为该国家所制作production_country_name 0/1
#最后消除production_countries和all_countries属性
#将production_countries分成num_countries和all_countries两个属性
train['num_countries'] = train['production_countries'].apply(lambda x: len(x) if x != {} else 0)
train['all_countries'] = train['production_countries'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
top_countries = [m[0] for m in Counter([i for j in list_of_countries for i in j]).most_common(25)]
for g in top_countries:
    train['production_country_' + g] = train['all_countries'].apply(lambda x: 1 if g in x else 0)
    
test['num_countries'] = test['production_countries'].apply(lambda x: len(x) if x != {} else 0)
test['all_countries'] = test['production_countries'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
for g in top_countries:
    test['production_country_' + g] = test['all_countries'].apply(lambda x: 1 if g in x else 0)

train = train.drop(['production_countries', 'all_countries'], axis=1)
test = test.drop(['production_countries', 'all_countries'], axis=1)
#分为了num_languages（语言版本数量）和all_languages（每部电影的所有语言版本）两个属性
#创建了30个新的属性，代表该电影是否有该语言的版本language_name 0/1
#最后消除spoken_languages和all_languages属性
#将spoken_languages属性分为num_languages（语言数量）和all_languages
train['num_languages'] = train['spoken_languages'].apply(lambda x: len(x) if x != {} else 0)
train['all_languages'] = train['spoken_languages'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
top_languages = [m[0] for m in Counter([i for j in list_of_languages for i in j]).most_common(30)]
for g in top_languages:
    train['language_' + g] = train['all_languages'].apply(lambda x: 1 if g in x else 0)
    
test['num_languages'] = test['spoken_languages'].apply(lambda x: len(x) if x != {} else 0)
test['all_languages'] = test['spoken_languages'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
for g in top_languages:
    test['language_' + g] = test['all_languages'].apply(lambda x: 1 if g in x else 0)

train = train.drop(['spoken_languages', 'all_languages'], axis=1)
test = test.drop(['spoken_languages', 'all_languages'], axis=1)
#分为了num_Keywords（ 每部电影关键词个数）和all_Keywords（每部电影的所有关键词）两个属性
#将Keywords分成num_Keywords（ 每部电影关键词个数）和all_Keywords
train['num_Keywords'] = train['Keywords'].apply(lambda x: len(x) if x != {} else 0)
train['all_Keywords'] = train['Keywords'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
#出现次数最多的关键词组成一个列表
top_keywords = [m[0] for m in Counter([i for j in list_of_keywords for i in j]).most_common(30)]
#将这些热度高的关键词各自组成一个属性（0、1）
for g in top_keywords:
    train['keyword_' + g] = train['all_Keywords'].apply(lambda x: 1 if g in x else 0)
    
test['num_Keywords'] = test['Keywords'].apply(lambda x: len(x) if x != {} else 0)
test['all_Keywords'] = test['Keywords'].apply(lambda x: ' '.join(sorted([i['name'] for i in x])) if x != {} else '')
for g in top_keywords:
    test['keyword_' + g] = test['all_Keywords'].apply(lambda x: 1 if g in x else 0)

train = train.drop(['Keywords', 'all_Keywords'], axis=1)
test = test.drop(['Keywords', 'all_Keywords'], axis=1)
#建立了num_cast（演员数量）的属性
#创建了15个新的属性，代表该电影是否有该演员cast_name_XXX 0/1
#增加性别的属性genders_0_cast genders_1_cast genders_2_cast，代表着这部电影不同性别的演员数量分别是多少。 
#最后消除cast属性
#将cast属性转为num_cast属性(演员数量)
train['num_cast'] = train['cast'].apply(lambda x: len(x) if x != {} else 0)
#出演次数最多的演员放在一个列表里
top_cast_names = [m[0] for m in Counter([i for j in list_of_cast_names for i in j]).most_common(15)]
#将这些演员单独变成一个属性（0，1）
for g in top_cast_names:
    train['cast_name_' + g] = train['cast'].apply(lambda x: 1 if g in str(x) else 0)
#增加3列属性，代表不同的性别
train['genders_0_cast'] = train['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))
train['genders_1_cast'] = train['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))
train['genders_2_cast'] = train['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))
#统计出演率最高的角色
top_cast_characters = [m[0] for m in Counter([i for j in list_of_cast_characters for i in j]).most_common(15)]
#将出演率最高的角色单独作为属性
for g in top_cast_characters:
    train['cast_character_' + g] = train['cast'].apply(lambda x: 1 if g in str(x) else 0)
    
test['num_cast'] = test['cast'].apply(lambda x: len(x) if x != {} else 0)
for g in top_cast_names:
    test['cast_name_' + g] = test['cast'].apply(lambda x: 1 if g in str(x) else 0)
test['genders_0_cast'] = test['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))
test['genders_1_cast'] = test['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))
test['genders_2_cast'] = test['cast'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))
for g in top_cast_characters:
    test['cast_character_' + g] = test['cast'].apply(lambda x: 1 if g in str(x) else 0)

train = train.drop(['cast'], axis=1)
test = test.drop(['cast'], axis=1)
#建立了num_crew（制作团队人员数量）的属性
#创建了15个新的属性，代表该电影是否有该制作团队人员crew_name_XXX 0/1
#增加性别的属性genders_0_crew genders_1_crew genders_2_crew，代表着这个电影团队不同性别的人员数量分别是多少。 
#每部电影制作团队的人数
train['num_crew'] = train['crew'].apply(lambda x: len(x) if x != {} else 0)
#最火的制作团队人员名字
top_crew_names = [m[0] for m in Counter([i for j in list_of_crew_names for i in j]).most_common(15)]
#将这些人单独设立一个属性
for g in top_crew_names:
    train['crew_name_' + g] = train['crew'].apply(lambda x: 1 if g in str(x) else 0)
#设立不同性别的制作团队人员数量属性
train['genders_0_crew'] = train['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))
train['genders_1_crew'] = train['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))
train['genders_2_crew'] = train['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))
#设立最主要的制作团队职务列表
top_crew_jobs = [m[0] for m in Counter([i for j in list_of_crew_jobs for i in j]).most_common(15)]
#单独设立属性
for j in top_crew_jobs:
    train['jobs_' + j] = train['crew'].apply(lambda x: sum([1 for i in x if i['job'] == j]))
#设立最主要的制作团队部分列表
top_crew_departments = [m[0] for m in Counter([i for j in list_of_crew_departments for i in j]).most_common(15)]
#单独设立属性
for j in top_crew_departments:
    train['departments_' + j] = train['crew'].apply(lambda x: sum([1 for i in x if i['department'] == j])) 

test['num_crew'] = test['crew'].apply(lambda x: len(x) if x != {} else 0)
for g in top_crew_names:
    test['crew_name_' + g] = test['crew'].apply(lambda x: 1 if g in str(x) else 0)
test['genders_0_crew'] = test['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 0]))
test['genders_1_crew'] = test['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 1]))
test['genders_2_crew'] = test['crew'].apply(lambda x: sum([1 for i in x if i['gender'] == 2]))
for j in top_crew_jobs:
    test['jobs_' + j] = test['crew'].apply(lambda x: sum([1 for i in x if i['job'] == j]))
for j in top_crew_departments:
    test['departments_' + j] = test['crew'].apply(lambda x: sum([1 for i in x if i['department'] == j])) 

train = train.drop(['crew'], axis=1)
test = test.drop(['crew'], axis=1)
#将年份进行丰富，比如18变成2018
def fix_date(x):
    """
    Fixes dates which are in 20xx
    """
    year = x.split('/')[2]
    if int(year) <= 19:
        return x[:-2] + '20' + year
    else:
        return x[:-2] + '19' + year
#循环提取日期信息并添加新列
def process_date(df):
    date_parts = ["year", "weekday", "month", 'day', 'quarter']
    for part in date_parts:
        part_col = 'release_date' + "_" + part
        df[part_col] = getattr(df['release_date'].dt, part).astype(int)
    
    # 处理 weekofyear
    df['release_date_weekofyear'] = df['release_date'].dt.isocalendar().week.astype(int)
    
    return df
#————————————————————————————————————————————————————————————————————————————————————————————



#创建新属性
train['log_revenue'] = np.log1p(train['revenue'])
train['log_budget'] = np.log1p(train['budget'])
test['log_budget'] = np.log1p(test['budget'])
#添加拥有官方页面的属性
train['has_homepage'] = 0
#train['homepage'].isnull()这部分代码会对 train 数据集中 homepage 列的每个元素进行检查，判断其是否为缺失值（NaN）。
# 返回一个布尔类型的 Series，其中 True 表示对应位置的元素是缺失值，False 表示不是缺失值。
#使用 loc 方法根据上述布尔 Series 进行索引，选取 homepage 列不为缺失值的那些行，并指定 has_homepage 列。
train.loc[train['homepage'].isnull() == False, 'has_homepage'] = 1
test['has_homepage'] = 0
test.loc[test['homepage'].isnull() == False, 'has_homepage'] = 1
#————————————————————————————————————————————————————————————————————————————————————————————
#TfidfVectorizer用于将文本数据转换为数值特征矩阵。
vectorizer = TfidfVectorizer(
            #采用次线性 tf（词频）缩放，即使用 1 + log(tf) 而不是 tf 来计算词频，这样可以减轻高频词的影响。
            sublinear_tf=True,
            #指定分析器为基于单词进行分析，即把文本拆分成单词来处理。
            analyzer='word',
            #定义分词模式，\w 表示匹配任何字母、数字或下划线字符，{1,} 表示匹配前面的字符至少一次，所以该模式会将文本按单词进行分割。
            token_pattern=r'\w{1,}',
            #指定 n - gram 的范围，这里表示同时考虑单个单词（unigram）和相邻两个单词组成的词组（bigram）作为特征。
            ngram_range=(1, 2),
            #设置最小文档频率为 5，即只有在至少 5 个文档中出现过的单词或词组才会被作为特征，这样可以过滤掉一些低频、可能是噪声的词汇。
            min_df=5)
#数据转换为 TF - IDF 特征矩阵
overview_text = vectorizer.fit_transform(train['overview'].fillna(''))
#LinearRegression()：创建一个线性回归模型对象。
linreg = LinearRegression()
#使用提取的文本特征矩阵 overview_text 作为输入特征，train['log_revenue'] 作为目标变量，对线性回归模型进行训练。
linreg.fit(overview_text, train['log_revenue'])


#发布日期或上映日期属性
#将前面选取的那些 release_date 列中的缺失值统一替换为 '01/01/98' 这个日期字符串
test.loc[test['release_date'].isnull() == True, 'release_date'] = '01/01/98'

#将数据集中年份进行丰富
train['release_date'] = train['release_date'].apply(lambda x: fix_date(x))
test['release_date'] = test['release_date'].apply(lambda x: fix_date(x))
#to_datetime将输入数据转换为 datetime 类型的函数。
train['release_date'] = pd.to_datetime(train['release_date'])
test['release_date'] = pd.to_datetime(test['release_date'])

# 应用处理函数
train = process_date(train)
test = process_date(test)

#删除不必要的列
train = train.drop(['homepage', 'imdb_id', 'poster_path', 'release_date', 'status', 'log_revenue'], axis=1)
test = test.drop(['homepage', 'imdb_id', 'poster_path', 'release_date', 'status'], axis=1)

for col in train.columns:
    if train[col].nunique() == 1:
        print(col)
        train = train.drop([col], axis=1)
        test = test.drop([col], axis=1)
        
        
for col in ['original_language', 'collection_name', 'all_genres']:
    #对 train 和 test 数据集中的 original_language、collection_name 和 all_genres 这三列分类特征进行编码转换
    # 将文本形式的类别标签转换为整数编码，以便后续的机器学习模型能够处理这些数据。
    le = LabelEncoder()
    #将 train 和 test 数据集中当前列的数据合并成一个列表。
    # 让 LabelEncoder 对象 le 学习这个合并列表中的所有唯一值，并为每个唯一值分配一个整数编码。
    le.fit(list(train[col].fillna('')) + list(test[col].fillna('')))#学习
    #le.transform(...) 使用之前拟合好的 LabelEncoder 对象 le 对 train 数据集中当前列的数据进行编码转换，将每个类别标签替换为对应的整数编码。
    train[col] = le.transform(train[col].fillna('').astype(str))#转换
    test[col] = le.transform(test[col].fillna('').astype(str))
    
train_texts = train[['title', 'tagline', 'overview', 'original_title']]
test_texts = test[['title', 'tagline', 'overview', 'original_title']]

for col in ['title', 'tagline', 'overview', 'original_title']:
    #从title这个属性分为了len_title（字符串长度）和words_title（单词个数）这两个属性
    train['len_' + col] = train[col].fillna('').apply(lambda x: len(str(x)))
    train['words_' + col] = train[col].fillna('').apply(lambda x: len(str(x.split(' '))))
    train = train.drop(col, axis=1)
    
    test['len_' + col] = test[col].fillna('').apply(lambda x: len(str(x)))
    test['words_' + col] = test[col].fillna('').apply(lambda x: len(str(x.split(' '))))
    test = test.drop(col, axis=1)
#对数据进行最后的修改
def endOfFix():
    # data fixes from https://www.kaggle.com/somang1418/happy-valentines-day-and-keep-kaggling-3
    train.loc[train['id'] == 16,'revenue'] = 192864          # Skinning
    train.loc[train['id'] == 90,'budget'] = 30000000         # Sommersby          
    train.loc[train['id'] == 118,'budget'] = 60000000        # Wild Hogs
    train.loc[train['id'] == 149,'budget'] = 18000000        # Beethoven
    train.loc[train['id'] == 313,'revenue'] = 12000000       # The Cookout 
    train.loc[train['id'] == 451,'revenue'] = 12000000       # Chasing Liberty
    train.loc[train['id'] == 464,'budget'] = 20000000        # Parenthood
    train.loc[train['id'] == 470,'budget'] = 13000000        # The Karate Kid, Part II
    train.loc[train['id'] == 513,'budget'] = 930000          # From Prada to Nada
    train.loc[train['id'] == 797,'budget'] = 8000000         # Welcome to Dongmakgol
    train.loc[train['id'] == 819,'budget'] = 90000000        # Alvin and the Chipmunks: The Road Chip
    train.loc[train['id'] == 850,'budget'] = 90000000        # Modern Times
    train.loc[train['id'] == 1112,'budget'] = 7500000        # An Officer and a Gentleman
    train.loc[train['id'] == 1131,'budget'] = 4300000        # Smokey and the Bandit   
    train.loc[train['id'] == 1359,'budget'] = 10000000       # Stir Crazy 
    train.loc[train['id'] == 1542,'budget'] = 1              # All at Once
    train.loc[train['id'] == 1570,'budget'] = 15800000       # Crocodile Dundee II
    train.loc[train['id'] == 1571,'budget'] = 4000000        # Lady and the Tramp
    train.loc[train['id'] == 1714,'budget'] = 46000000       # The Recruit
    train.loc[train['id'] == 1721,'budget'] = 17500000       # Cocoon
    train.loc[train['id'] == 1865,'revenue'] = 25000000      # Scooby-Doo 2: Monsters Unleashed
    train.loc[train['id'] == 2268,'budget'] = 17500000       # Madea Goes to Jail budget
    train.loc[train['id'] == 2491,'revenue'] = 6800000       # Never Talk to Strangers
    train.loc[train['id'] == 2602,'budget'] = 31000000       # Mr. Holland's Opus
    train.loc[train['id'] == 2612,'budget'] = 15000000       # Field of Dreams
    train.loc[train['id'] == 2696,'budget'] = 10000000       # Nurse 3-D
    train.loc[train['id'] == 2801,'budget'] = 10000000       # Fracture
    test.loc[test['id'] == 3889,'budget'] = 15000000       # Colossal
    test.loc[test['id'] == 6733,'budget'] = 5000000        # The Big Sick
    test.loc[test['id'] == 3197,'budget'] = 8000000        # High-Rise
    test.loc[test['id'] == 6683,'budget'] = 50000000       # The Pink Panther 2
    test.loc[test['id'] == 5704,'budget'] = 4300000        # French Connection II
    test.loc[test['id'] == 6109,'budget'] = 281756         # Dogtooth
    test.loc[test['id'] == 7242,'budget'] = 10000000       # Addams Family Values
    test.loc[test['id'] == 7021,'budget'] = 17540562       #  Two Is a Family
    test.loc[test['id'] == 5591,'budget'] = 4000000        # The Orphanage
    test.loc[test['id'] == 4282,'budget'] = 20000000       # Big Top Pee-wee

    power_six = train.id[train.budget > 1000][train.revenue < 100]

    for k in power_six :
        train.loc[train['id'] == k,'revenue'] =  train.loc[train['id'] == k,'revenue'] * 1000000

endOfFix()
        
X = train.drop(['id', 'revenue'], axis=1)

# 打开文件以写入模式
with open('revenue.txt', 'w') as file:
    # 遍历 revenue 列的每个值
    for value in train['revenue']:
        # 将值转换为字符串并写入文件，然后添加换行符
        file.write(str(value) + '\n')

print("数据已成功写入 revenue.txt 文件。")
y = np.log1p(train['revenue'])
X_test = test.drop(['id'], axis=1)
#————————————————————————————————————————————————————————————————————————————————————————————   
#————————————————————————————————————————————————————————————————————————————————————————————
#模型建立

#test_size=0.1：这个参数指定了验证集在整个数据集中所占的比例。这里 test_size = 0.1 表示将数据集的 10% 作为验证集，剩下的 90% 作为训练集。
#X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1)
#在 K 折交叉验证中，n_fold 表示要将数据集划分成的折数。也就是说，后续会把数据集平均分成 10 个部分。
#交叉验证是一种常用的模型评估方法，它可以更有效地利用数据集，减少模型评估的偏差。
n_fold = 10
folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)

def train_model(X, X_test, y, params=None, folds=folds, model_type='lgb', plot_feature_importance=False, model=None):
    #oof：用于存储验证集上的预测结果，初始化为全零数组，长度为训练集的样本数。
    oof = np.zeros(X.shape[0])
    #prediction：用于存储测试集上的预测结果，初始化为全零数组，长度为测试集的样本数。
    prediction = np.zeros(X_test.shape[0])
    #scores：用于存储每一折的评估分数。
    scores = []
    #feature_importance：用于存储特征重要性信息，初始化为一个空的 DataFrame。
    feature_importance = pd.DataFrame()
    
    #根据 model_type 的不同，选择合适的方式划分训练集和验证集。
    # 如果是 'sklearn' 模型，直接使用索引切片；否则，先将 X 转换为 numpy 数组再进行切片。
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print('Fold', fold_n, 'started at', time.ctime())
        if model_type == 'sklearn':
            X_train, X_valid = X[train_index], X[valid_index]
        else:
            X_train, X_valid = X.values[train_index], X.values[valid_index]
            
        y_train, y_valid = y[train_index], y[valid_index]
        
        if model_type == 'lgb':
            model = lgb.LGBMRegressor(**params, n_estimators = 20000, nthread = 4, n_jobs = -1,verbose=1000)
            callbacks = [early_stopping(stopping_rounds=200)]
            model.fit(X_train, y_train, 
                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='rmse', callbacks=callbacks)
            
            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
            
        if model_type == 'xgb':
            train_data = xgb.DMatrix(data=X_train, label=y_train)
            valid_data = xgb.DMatrix(data=X_valid, label=y_valid)

            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=500, params=params)
            y_pred_valid = model.predict(xgb.DMatrix(X_valid), iteration_range=(0, model.best_iteration + 1))
            y_pred = model.predict(xgb.DMatrix(X_test.values), iteration_range=(0, model.best_iteration + 1))

        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)
            y_pred_valid = model.predict(X_valid).reshape(-1,)
            score = mean_squared_error(y_valid, y_pred_valid)
            
            y_pred = model.predict(X_test)
            
        if model_type == 'cat':
            model = CatBoostRegressor(iterations=20000,  eval_metric='RMSE', **params)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)

            y_pred_valid = model.predict(X_valid)
            y_pred = model.predict(X_test)
        
        oof[valid_index] = y_pred_valid.reshape(-1,)
        scores.append(mean_squared_error(y_valid, y_pred_valid) ** 0.5)
        
        prediction += y_pred    
        
        if model_type == 'lgb':
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = X.columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= n_fold
    
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    
    if model_type == 'lgb':
        feature_importance["importance"] /= n_fold
        if plot_feature_importance:
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGB Features (avg over folds)');
        
            return oof, prediction, feature_importance,model
        return oof, prediction,model
    
    else:
        return oof, prediction,model

for col in train_texts.columns:
    vectorizer = TfidfVectorizer(
                sublinear_tf=True,
                analyzer='word',
                token_pattern=r'\w{1,}',
                ngram_range=(1, 2),
                min_df=10
    )
    vectorizer.fit(list(train_texts[col].fillna('')) + list(test_texts[col].fillna('')))
    train_col_text = vectorizer.transform(train_texts[col].fillna(''))
    test_col_text = vectorizer.transform(test_texts[col].fillna(''))
    #创建一个带交叉验证的岭回归模型对象。
    #alphas=(0.01, 0.1, 1.0, 10.0, 100.0)：指定要尝试的正则化参数  的取值范围。
    #scoring='neg_mean_squared_error'：指定评估指标为负均方误差
    #cv=folds：指定交叉验证的折数和划分方式，folds 应该是一个 KFold 或其他交叉验证对象。
    model = linear_model.RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0), scoring='neg_mean_squared_error', cv=folds)
    oof_text, prediction_text,text_model = train_model(train_col_text, test_col_text, y, params=None, model_type='sklearn', model=model)
    
    X[col + '_oof'] = oof_text
    X_test[col + '_oof'] = prediction_text
    
def new_features(df):
    df['budget_to_popularity'] = df['budget'] / df['popularity']
    df['budget_to_runtime'] = df['budget'] / df['runtime']
    
    # some features from https://www.kaggle.com/somang1418/happy-valentines-day-and-keep-kaggling-3
    df['_budget_year_ratio'] = df['budget'] / (df['release_date_year'] * df['release_date_year'])
    df['_releaseYear_popularity_ratio'] = df['release_date_year'] / df['popularity']
    df['_releaseYear_popularity_ratio2'] = df['popularity'] / df['release_date_year']
    
    df['runtime_to_mean_year'] = df['runtime'] / df.groupby("release_date_year")["runtime"].transform('mean')
    df['popularity_to_mean_year'] = df['popularity'] / df.groupby("release_date_year")["popularity"].transform('mean')
    df['budget_to_mean_year'] = df['budget'] / df.groupby("release_date_year")["budget"].transform('mean')
        
    return df

X = new_features(X)
X_test = new_features(X_test)    
    
def top_cols_interaction(df):
    df['budget_to_year'] = df['budget'] / df['release_date_year']
    df['budget_to_mean_year_to_year'] = df['budget_to_mean_year'] / df['release_date_year']
    df['popularity_to_mean_year_to_log_budget'] = df['popularity_to_mean_year'] / df['log_budget']
    df['year_to_log_budget'] = df['release_date_year'] / df['log_budget']
    df['budget_to_runtime_to_year'] = df['budget_to_runtime'] / df['release_date_year']
    df['genders_1_cast_to_log_budget'] = df['genders_1_cast'] / df['log_budget']
    df['all_genres_to_popularity_to_mean_year'] = df['all_genres'] / df['popularity_to_mean_year']
    df['genders_2_crew_to_budget_to_mean_year'] = df['genders_2_crew'] / df['budget_to_mean_year']
    df['overview_oof_to_genders_2_crew'] = df['overview_oof'] / df['genders_2_crew']
    
    return df

X = top_cols_interaction(X)
X_test = top_cols_interaction(X_test)

X = X.replace([np.inf, -np.inf], 0).fillna(0)
X_test = X_test.replace([np.inf, -np.inf], 0).fillna(0)

trainAdditionalFeatures = pd.read_csv('MoviesData/rawPredictData/TrainAdditionalFeatures.csv')
testAdditionalFeatures = pd.read_csv('MoviesData/rawPredictData/TestAdditionalFeatures.csv')

#增加imdb_id这个属性用于连接属性
train = pd.read_csv('MoviesData/rawPredictData/train.csv')
test = pd.read_csv('MoviesData/rawPredictData/test.csv')
X['imdb_id'] = train['imdb_id']
X_test['imdb_id'] = test['imdb_id']
del train, test

#将新增的属性合并
X = pd.merge(X, trainAdditionalFeatures, how='left', on=['imdb_id'])
X_test = pd.merge(X_test, testAdditionalFeatures, how='left', on=['imdb_id'])

X = X.drop(['imdb_id'], axis=1)
X_test = X_test.drop(['imdb_id'], axis=1)

import mysql.connector
db_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': 'root',
    'database': 'project'
}

try:
    # 打开文件，指定编码为 utf-8
    with open("output.txt", 'w', encoding='utf-8') as file:
        # 写入列名
        column_names = '\t'.join(X.columns)
        file.write(column_names + '\n')

        # 遍历每一行
        for _, row in X.iterrows():
            # 将每行数据转换为字符串，用制表符连接
            row_data = '\t'.join(str(value) for value in row)
            # 写入一行数据并换行
            file.write(row_data + '\n')
    print(f"数据已成功保存到 {'output.txt'}")
except Exception as e:
    print(f"保存数据时出现错误: {e}")

#——————————————————————————————————————————————————————————————————————————————————
#预测阶段
params = {'num_leaves': 30,
         'min_data_in_leaf': 20,
         'objective': 'regression',
         'max_depth': 9,
         'learning_rate': 0.01,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.2,
         "verbosity": -1}
oof_lgb, prediction_lgb, _ ,lgb_model= train_model(X, X_test, y, params=params, model_type='lgb', plot_feature_importance=True)
dump(lgb_model, 'Rawmodel1.joblib')

xgb_params = {'eta': 0.01,
              'objective': 'reg:linear',
              'max_depth': 7,
              'subsample': 0.8,
              'colsample_bytree': 0.8,
              'eval_metric': 'rmse',
              'seed': 11,
              'silent': True}
oof_xgb, prediction_xgb , xgb_model= train_model(X, X_test, y, params=xgb_params, model_type='xgb', plot_feature_importance=False)
dump(xgb_model, 'Rawmodel2.joblib')

cat_params = {'learning_rate': 0.002,
              'depth': 5,
              'l2_leaf_reg': 10,
              # 'bootstrap_type': 'Bernoulli',
              'colsample_bylevel': 0.8,
              'bagging_temperature': 0.2,
              #'metric_period': 500,
              'od_type': 'Iter',
              'od_wait': 100,
              'random_seed': 11,
              'allow_writing_files': False}
oof_cat, prediction_cat, cat_model = train_model(X, X_test, y, params=cat_params, model_type='cat')
dump(cat_model, 'Rawmodel3.joblib')


params = {'num_leaves': 30,
         'min_data_in_leaf': 20,
         'objective': 'regression',
         'max_depth': 5,
         'learning_rate': 0.01,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.2,
         "verbosity": -1}
oof_lgb_1, prediction_lgb_1, lgb1_model = train_model(X, X_test, y, params=params, model_type='lgb', plot_feature_importance=False)
dump(lgb1_model, 'Rawmodel4.joblib')

params = {'num_leaves': 30,
         'min_data_in_leaf': 20,
         'objective': 'regression',
         'max_depth': 7,
         'learning_rate': 0.02,
         "boosting": "gbdt",
         "feature_fraction": 0.7,
         "bagging_freq": 5,
         "bagging_fraction": 0.7,
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.2,
         "verbosity": -1}
oof_lgb_2, prediction_lgb_2,lgb2_model = train_model(X, X_test, y, params=params, model_type='lgb', plot_feature_importance=False)
dump(lgb2_model, 'Rawmodel5.joblib')


#拼接起来
train_stack = np.vstack([oof_lgb, oof_xgb, oof_cat, oof_lgb_1, oof_lgb_2]).transpose()
train_stack = pd.DataFrame(train_stack, columns=['lgb', 'xgb', 'cat', 'lgb_1', 'lgb_2'])
test_stack = np.vstack([prediction_lgb, prediction_xgb, prediction_cat, prediction_lgb_1, prediction_lgb_2]).transpose()
test_stack = pd.DataFrame(test_stack, columns=['lgb', 'xgb', 'cat', 'lgb_1', 'lgb_2'])



params = {'num_leaves': 8,
         'min_data_in_leaf': 20,
         'objective': 'regression',
         'max_depth': 3,
         'learning_rate': 0.01,
         "boosting": "gbdt",
         "bagging_seed": 11,
         "metric": 'rmse',
         "lambda_l1": 0.2,
         "verbosity": -1}
oof_lgb_stack, prediction_lgb_stack, _ , lgb_stack_model= train_model(train_stack, test_stack, y, params=params, model_type='lgb', plot_feature_importance=True)

dump(lgb_stack_model, 'lightgbm_model.joblib')

