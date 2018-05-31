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

k-means 算法将一组 ![N](http://sklearn.apachecn.org/cn/0.19.0/_images/math/f4170ed8938b79490d8923857962695514a8e4cb.png) 样本 ![X](http://sklearn.apachecn.org/cn/0.19.0/_images/math/7a7bb470119808e2db2879fc2b2526f467b7a40b.png) 划分成 ![K](http://sklearn.apachecn.org/cn/0.19.0/_images/math/684381a21cd73ebbf43b63a087d3f7410ee99ce8.png) 不相交的 簇 ![C](http://sklearn.apachecn.org/cn/0.19.0/_images/math/afce44aa7c55836ca9345404c22fc7b599d2ed84.png), 每个都用该簇中的样本的均值 ![\mu_j](http://sklearn.apachecn.org/cn/0.19.0/_images/math/169afc05a24e52e428b94e0041ab0577a2d580ee.png) 描述。 这个均值通常被称为簇的 “质心”; 注意，它们一般不是从 ![X](http://sklearn.apachecn.org/cn/0.19.0/_images/math/7a7bb470119808e2db2879fc2b2526f467b7a40b.png) 中挑选出的点，虽然它们是处在同一个 空间。 K-means算法旨在选择最小化惯性或 簇内和的平方和的标准的质心:

![\sum_{i=0}^{n}\min_{\mu_j \in C}(||x_j - \mu_i||^2)](http://sklearn.apachecn.org/cn/0.19.0/_images/math/1886f2c69775746ac7b6c1cdd88c53c676839015.png)

K-means 通常被称为劳埃德算法。在基本术语中，算法有三个步骤。、 第一步是选择初始质心，最基本的方法是从 ![X](http://sklearn.apachecn.org/cn/0.19.0/_images/math/7a7bb470119808e2db2879fc2b2526f467b7a40b.png) 数据集中选择 ![k](http://sklearn.apachecn.org/cn/0.19.0/_images/math/0b7c1e16a3a8a849bb8ffdcdbf86f65fd1f30438.png) 个样本。初始化完成后，K-means 由两个其他步骤之间的循环组成。 第一步将每个样本分配到其 最近的质心。第二步通过取分配给每个先前质心的所有样本的平均值来创建新的质心。计算旧的和新的质心之间的差异，并且算法重复这些最后的两个步骤，直到该值小于阈值。换句话说，算法重复这个步骤，直到质心不再显著移动。

给定足够的时间，K-means 将总是收敛的，但这可能是 局部最优的。这很大程度上取决于质心的初始化。 因此，通常会进行几次初始化不同质心的计算。帮助解决这个问题的一种方法是 k-means++ 初始化方案。 这将初始化 质心通常彼此远离，导致比随机初始化更好的结果。

#### 实现过程

#### 簇类数目确定

##### 轮廓(Silhouette)系数

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

###### 优点

- 对于不正确的 clustering （聚类），分数为 -1 ， highly dense clustering （高密度聚类）为 +1 。零点附近的分数表示 overlapping clusters （重叠的聚类）。
- 当 clusters （簇）密集且分离较好时，分数更高，这与 cluster （簇）的标准概念有关。

######  缺点

- convex clusters（凸的簇）的 Silhouette Coefficient 通常比其他类型的 cluster （簇）更高，例如通过 DBSCAN 获得的基于密度的 cluster（簇）。

##### Calinski-Harabaz 指数

如果不知道真实数据的类别标签，则可以使用 Calinski-Harabaz 指数来评估模型，其中较高的 Calinski-Harabaz 的得分与具有更好定义的聚类的模型相关。

对于 ![k](http://sklearn.apachecn.org/cn/0.19.0/_images/math/0b7c1e16a3a8a849bb8ffdcdbf86f65fd1f30438.png) 簇，Calinski-Harabaz 得分 ![s](http://sklearn.apachecn.org/cn/0.19.0/_images/math/63751cb2e98ba393b0f22e45ca127c3cebb61487.png) 是作为 between-clusters dispersion mean （簇间色散平均值）与 within-cluster dispersion（群内色散之间）的比值给出的:

![s(k) = \frac{\mathrm{Tr}(B_k)}{\mathrm{Tr}(W_k)} \times \frac{N - k}{k - 1}](http://sklearn.apachecn.org/cn/0.19.0/_images/math/66e217c045c57898975dd3d3ea651747ed9a5c19.png)

其中 ![B_K](http://sklearn.apachecn.org/cn/0.19.0/_images/math/b0cd669148609abb7c9af6fa3e706b7b79577b5c.png) 是 between group dispersion matrix （组间色散矩阵）， ![W_K](http://sklearn.apachecn.org/cn/0.19.0/_images/math/e7d8801b1f41dc013f994d181b7826d2a0fc4f88.png) 是由以下定义的 within-cluster dispersion matrix （群内色散矩阵）:

![W_k = \sum_{q=1}^k \sum_{x \in C_q} (x - c_q) (x - c_q)^T](http://sklearn.apachecn.org/cn/0.19.0/_images/math/48eba4dc277d1cbf5d1f61fe7ec36042198b7a98.png)

![B_k = \sum_q n_q (c_q - c) (c_q - c)^T](http://sklearn.apachecn.org/cn/0.19.0/_images/math/488a40c7485d836c31ddf1b5d4267429d625983e.png)

![N](http://sklearn.apachecn.org/cn/0.19.0/_images/math/f4170ed8938b79490d8923857962695514a8e4cb.png) 为数据中的点数，![C_q](http://sklearn.apachecn.org/cn/0.19.0/_images/math/98a0fb38d49709c39a35007dd817dff8b7b3e68a.png) 为 cluster （簇） ![q](http://sklearn.apachecn.org/cn/0.19.0/_images/math/620a3ce6403ec82f1347af9985bc03f7a9382f4a.png) 中的点集， ![c_q](http://sklearn.apachecn.org/cn/0.19.0/_images/math/70819c4bdcf3aecea24eac192c0365fa0ccab488.png) 为 cluster（簇） ![q](http://sklearn.apachecn.org/cn/0.19.0/_images/math/620a3ce6403ec82f1347af9985bc03f7a9382f4a.png) 的中心， ![c](http://sklearn.apachecn.org/cn/0.19.0/_images/math/ae12a24f88803b5895632e4848d87d46483c492c.png) 为 ![E](http://sklearn.apachecn.org/cn/0.19.0/_images/math/4b6222b865b812d2a59368cd1629eed6b54454d5.png) 的中心， ![n_q](http://sklearn.apachecn.org/cn/0.19.0/_images/math/435c528d448d9b4bdaf384010cede06da9c69c32.png) 为 cluster（簇） ![q](http://sklearn.apachecn.org/cn/0.19.0/_images/math/620a3ce6403ec82f1347af9985bc03f7a9382f4a.png) 中的点数。

###### 优点

- 当 cluster （簇）密集且分离较好时，分数更高，这与一个标准的 cluster（簇）有关。
- 得分计算很快

###### 缺点

- 凸的簇的 Calinski-Harabaz index（Calinski-Harabaz 指数）通常高于其他类型的 cluster（簇），例如通过 DBSCAN 获得的基于密度的 cluster（簇）。

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

### 文本聚类

#### 簇类数目选取

![](/home/whao/Desktop/kzhi.png)

#### 运行时间

![](/home/whao/Desktop/time.png)

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







