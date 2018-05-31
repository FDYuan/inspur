# Inspur 浪潮实习生挑战赛

## 题目需求

需要建立**新闻数据库**，设计实现基于该数据库的新闻**内容分类**和**推荐算法**，并做相应的**前端效果展示**。

## 设计方案

###程序设计流程

![流程图](/home/whao/Downloads/流程图.svg)

其操作步骤为：

0. 爬虫爬取的（）网的新闻信息并存入外部文件
1. 将外部文件处理后存入数据库，使用SQLite3
2. 从数据库提取新闻并分词，使用jieba
3. 计算TF-IDF并使用K-means聚类（使用Naive Bayesian算法分类）
4. 根据用户特征向量计算余弦相似度

### 部分模块设计流程

以下是关键模块——文本聚类和新闻推荐的操作步骤。

文本聚类：

1. 计算文档分词得到词袋空间VSM，使用sklearn.CountVectorizer
2. 根据词袋空间计算TF-IDF，使用sklearn.TfidfTransformer
3. 利用TF-IDF完成K-means聚类，使用sklearn.Kmeans

新闻推荐：

1. 根据用户的特征标签和VSM计算用户的特征向量
2. 特征向量与VSM计算余弦相似度，使用numpy
3. 根据权值函数计算每个新闻的权值，排序并推荐

## 程序模块设计

### *网络爬虫*

#### 实现方式

使用request和beautifulsoup4库访问网站页面

页面URL根据其特征遍历得出。

### 分词

#### 关于分词

常见的中文分词方式有：jieba分词，中科院分词，百度分词。由于jieba分词对于python的良好兼容，以及其良好效果，本程序使用jieba分词。

#### 实现方式

引用python库jieba

```python
import jieba
seg_list = jieba.cut(input_text)
print(", ".join(seg_list))
```

### 文本聚类

#### 关于K-means算法

事先确定常数K，常数K意味着最终的聚类类别数，首先随机选定初始点为质心，并通过计算每一个样本与质心之间的相似度(这里为欧式距离)，将样本点归到最相似的类中，接着，重新计算每个类的质心(即为类中心)，重复这样的过程，知道质心不再改变，最终就确定了每个样本所属的类别以及每个类的质心。

算法的时间复杂度上界为`O(n*k*t)`, 其中t是迭代次数。

#### 实现方式

定义sklearn的Kmeans模型，然后将计算好的TF-IDF矩阵传入模型产生结果。

#### 簇类数目选择

根据计算轮廓系数确定。

1. 计算样本i到同簇其他样本的平均距离ai。ai 越小，说明样本i越应该被聚类到该簇。将ai 称为样本i的**簇内不相似度**。

   **簇C中所有样本的a i 均值称为簇C的簇不相似度。**

2. 计算样本i到其他某簇Cj 的所有样本的平均距离bij，称为样本i与簇Cj 的不相似度。定义为样本i的**簇间不相似度**：bi =min{bi1, bi2, ..., bik} 

   **bi越大，说明样本i越不属于其他簇。**

3. 根据样本i的簇内不相似度a i 和簇间不相似度b i ，定义样本i的**轮廓系数**：

   ![img](https://img-blog.csdn.net/20160720114532744)

4. 判断：

   -  si接近1，则说明样本i聚类合理；
   -  si接近-1，则说明样本i更应该分类到另外的簇；
   - 若si 近似为0，则说明样本i在两个簇的边界上。

 **所有样本的s i 的均值称为聚类结果的轮廓系数**，是该聚类是否合理、有效的度量。

### 文本分类(X)

#### 关于Naive Bayesian算法(X)

#### 实现方式(X)

### 推荐算法

#### 关于推荐算法

常见的推荐算法有：基于内容的推荐算法、协同过滤推荐算法。基于内容的推荐算法是根据用户过去喜欢的物品，为用户推荐和他过去喜欢的物品相似的物品。协同过滤推荐算法是通过对用户历史行为数据的挖掘发现用户的偏好，基于不同的偏好对用户进行群组划分并推荐品味相似的商品。

由于对用户历史行为信息的缺乏，本程序采用基于内容的推荐算法。

#### 实现方式

其主要思想为将用户特征值作为特征向量，计算用户特征向量于各个新闻之间的余弦相似度。考虑到新闻信息中有点击量和日期的存在，推荐K值采用加权公式的方式计算，算式为：
$$
K=0.7*cos\beta+0.2*P(X=k)+0.1*(n/m)
$$

#### 算式解读

##### 余弦相似度计算

[余弦相似度]: https://zh.wikipedia.org/zh-hans/%E4%BD%99%E5%BC%A6%E7%9B%B8%E4%BC%BC%E6%80%A7

$$
cos\beta =\sum_{1}^{n} \left ( a_{i}*b_{i} \right )/\sqrt{\sum_{1}^{n}a_{i}^2]}*\sqrt{\sum_{1}^{n}b_{i}^2]}
$$

###### 日期权重计算

[Goodle 趋势]: https://trends.google.com/trends/

假设新闻点击为独立事件。计算一段时间内新闻的点击量，其符合**泊松分布**。根据 Google 趋势中新闻点击量的统计，我们可以知道泊松分布模型适应大部分情况。而且观察新闻点击的分布区间，我们可以一般认为该样本的新闻时效性为**两天**。

###### 点击量权重计算(X)

根据数据库中的样本计算可以知道点击量同样满足**泊松分布**。



泊松分布公式为：
$$
P(X=k)=λ^k*e^-λ/k!~~,~~k=0,1,...
$$


### 前端和后端设计(X)

前端采用bootstrap设计，后端采用django框架设计

#### 目录结构(X)

#### 架构交互图(X)

## 数据接口设计

#### 数据库结构设计

data.db

|  属性   |     类型     |
| :-----: | :----------: |
|   id    |   integer    |
|  title  | varchar(255) |
| content |     text     |
| clicks  |   integer    |
|  date   |     text     |
|  type   |   integer    |

## 系统运行测试

### 依赖项

本系统依赖Python2.7运行环境并需要安装以下外部库：

1. beautifulsoup4
2. jieba
3. scipy
4. sklearn
5. matplotlib

安装方法为：

```bash
sudo pip install {name}
```

### 网络爬虫

#### 运行结果

![](/home/whao/Desktop/TIM图片20180531081620.png)

#### 性能评估

![](/home/whao/Desktop/TIM图片20180531081605.png)

###新闻分词

#### 运行结果

![分词展示](/home/whao/Desktop/深度截图_选择区域_20180531040659.png)

### 文本聚类(X)

#### 运行结果(X)

#### 性能评估(X)

### 文本分类(X)

#### 运行结果(X)

#### 性能评估(X)

### 推荐展示(X)

#### 运行结果(X)

#### 前端展示(X)

#### 性能评估(X)

## 核心代码

### 网络爬虫



```python
today = datetime.date.today()    #获取当前时间
times = 100   #要爬取的天数
ago=600       #从600天以前的信息开始爬
for i in range(times):
    dateadd = datetime.timedelta(days=i+ago)
    olddate = (today - dateadd).strftime('%Y/%m%d')
    url = "http://www.chinanews.com/scroll-news/" + olddate + "/news.shtml"   #自动更新链接中的日期
    print("已获取url："+url + "---------------")
    soup = delete_Script(getSoup(url))#获取一个BeautifulSoup对象

    mylist_class.extend(soup.findAll("div", class_="dd_lm"))# 类别标签
    mylist_title.extend(soup.findAll("div", class_="dd_bt"))# 链接标签

countt = len(mylist_class)
for j in range(countt):
    tempurl = mylist_title[j].a["href"]#查找相应标签中的url
    mylist_url.append(tempurl)#添加到列表中
    try:
        if re.compile(r'http').findall(str(mylist_url[k])):
            mysoup = delete_Script(getSoup(str(mylist_url[k])))#获取内容标签的BeautifulSoup
        else:
            mysoup = delete_Script(getSoup("http://www.chinanews.com" + str(mylist_url[k])))
        content = mysoup.find("div", "left_zw").text.strip() #获取正文内容
    except:
        continue
    writer.writerow((mylist_class[k].text, mylist_title[k].text, mylist_url[k], content))#将数据写到csv文件中
```

## 存在的问题(X)







