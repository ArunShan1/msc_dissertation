import numpy 
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import sklearn.cluster
import scipy
from yahmm import *

#Upload file from same folder
def upload_new(file):
    df = pd.read_csv(file,header=None,sep='\t')
    df.columns = ['hour','lang','posts','ufoll','ufriend','ustatus']
    df.loc[:,['hour']]=df.loc[:,['hour']].apply(lambda x: pd.to_datetime(x,format="%Y-%m-%d %H"))
    df.loc[df['lang'].isin(["msa","id"]), 'lang'] = 'ms'
    df['lang'] = df['lang'].astype('category')
    df = df.dropna(how='any')
    return(df)

#Create Table summarising Data
def summarytable(df,nlang):
    counts = df.groupby('lang').sum()
    counts = counts.sort_values(['posts'],ascending=False)
    counts['percentage']=100*counts['posts']/counts['posts'].sum()
    counts['avefoll']=counts['ufoll']/counts['posts']
    counts['avefriend']=counts['ufriend']/counts['posts']
    counts['foll/friend ratio']=counts['ufoll']/counts['ufriend']
    counts = counts.drop(['ufoll','ufriend','ustatus'], 1)
    counts = counts.head(nlang).round({'percentage':2, 'avefoll':0,'avefriend':0,'foll/friend ratio':2})
    counts[['avefoll','avefriend']] = counts[['avefoll','avefriend']].astype(int)
    return(counts)

#Get timeseries for multiple languages of proportional changes for a given feature
def tablegrowth(df,lolang,feature,foo):
    df['day']=df['hour'].dt.floor('d')
    counts = pd.pivot_table(df,values=feature,index='day',columns='lang',aggfunc=foo,margins=True)
    counts = counts[lolang+['All']]
    counts = counts.div(counts['All'],axis=0)
    counts = counts.drop(['All'],axis=1)
    counts = counts.drop(['All'],axis=0)
    counts = counts.pct_change(periods=1)
    counts = counts.dropna(axis=0,how='all')
    return(counts)

#Plot result of tablegrowth
def plotgrowth(table,titlename,titlesize,size):
    plot = table.plot(kind='line',legend=True,title=titlename,figsize=(15,10), fontsize=size)
    plot.set_ylabel("Growth", fontsize=size)
    plot.set_xlabel("Time", fontsize=size)
    plot.title.set_size(titlesize)
    plt.show()
    return

#Create a directory of databases with growth points for each feature of where each database is for a specific language
def points(lolang,df):
    tableposts = tablegrowth(df,lolang,'posts',numpy.sum)
    tablefoll = tablegrowth(df,lolang,'ufoll',numpy.sum)
    tablefriend = tablegrowth(df,lolang,'ufriend',numpy.sum)
    frames = {}
    for lang in lolang:
        frames[lang] = pd.concat([tableposts[lang], tablefoll[lang],tablefriend[lang]], axis=1, 
                            keys=['posts', 'foll','friend'])
    return(frames)

#Cluster using K-Means Clustering (Change for Agglomerative when necessary)
def cluster(lang,n,lofeatures,df):
    frames = points([lang],df)
    dataset=frames[lang][lofeatures]
    mat = dataset.as_matrix()
    km = sklearn.cluster.KMeans(n_clusters=n)
    km.fit(mat)
    labels = km.labels_
    dataset['label']=pd.Series(labels).values
    return(dataset)

#Do clustering except with labels adjusted for surge, growth, stabilise, decline labels.
def labeladjust(lang,n,lofeatures,feature,df):
    table = cluster(lang,n,lofeatures,df)
    d = [0]*n
    for i in range(0,n):
        d[i] = (i,min(table.loc[table['label'] == i,feature]))
    order = sorted(d, key=lambda x:x[1])
    dic = {}
    for i in range(0,n):
        dic[order[i][0]] = i
    table['label'] = table['label'].replace(dic)
    return(table)

#Get breaks for a specific feature within clustering where they separate
def getbreaks(lang,n,lofeatures,feature,df):
    table = cluster(lang,n,lofeatures,df)
    minmax = [0]*n
    for i in range(0,n):
        minmax[i] = (min(table.loc[table['label'] == i,feature]),max(table.loc[table['label'] == i,feature]))
    breaks = [0]*(n-1)
    ordered = sorted(minmax)
    for i in range(0,n-1):
        breaks[i] = numpy.mean([ordered[i][1],ordered[(i+1)][0]])
    return(breaks)

#Get probabilities of each stage occuring
def getproportions(lang,n,lofeatures,feature,df):
    table = labeladjust(lang,n,lofeatures,feature,df)
    return(table['label'].value_counts(normalize=True).sort_index())

#Get original transition matrix based on clustering
def transitionmatrix(lang,n,lofeatures,feature,df):
    table = labeladjust(lang,n,lofeatures,feature,df)
    table['nextlabel']=table['label'].shift(-1)
    table = table.dropna(how='any',axis=0)
    table['nextlabel'] = table['nextlabel'].astype(int)
    transition = table.groupby(['label','nextlabel']).size().unstack(level='nextlabel',fill_value=0)
    transition = transition.apply(lambda x: x/float(x.sum()),axis=1)
    return(transition)

#Run Hidden Markov Model to learn form clustered data
def hmmtransition(lang,n,lofeatures,feature,df):
    table = labeladjust(lang,n,lofeatures,feature,df)
    starts = list(table['label'].value_counts(normalize=True).sort_index())
    table['nextlabel']=table['label'].shift(-1)
    table = table.dropna(how='any',axis=0)
    table['nextlabel'] = table['nextlabel'].astype(int)
    transition = table.groupby(['label','nextlabel']).size().unstack(level='nextlabel',fill_value=0)
    transition = transition.apply(lambda x: x/float(x.sum()),axis=1)
    matrix = transition.as_matrix()
    means = table.groupby('label')['posts'].mean()
    stds = table.groupby('label')['posts'].std()
    distributions = [NormalDistribution(means[i],stds[i]) for i in range(n)]
    ends = [0] * n
    state_names= list(map(str,range(n)))
    model = Model.from_matrix( matrix, distributions, starts, ends,state_names, name="main_model" )
    sequence = list(table[feature])
    model.train([sequence], algorithm='baum-welch',stop_threshold=10)
    return(model)

#Get Supnorm of two matrices
def supnorm(mat1,mat2):
    x = mat1 - mat2
    y = x.as_matrix()
    return(max(y.min(), y.max(), key=abs))

#Get table showing distances of languages from each other
langs = ['en','ja','es','ar','pt','ms','ko','tr','th','fr','nl','tl','it']
direct = {}
for i in range(numpy.size(langs)):
    model = hmmtransition(langs[i],4,['posts','foll','friend'],'posts',df)
    direct[i] = pd.DataFrame(numpy.exp(model.dense_transition_matrix())[0:-2, 0:-2]).round(3)
a = [[supnorm(direct[x],direct[y]) for y in range(len(direct))] for x in range(len(direct))]
a = pd.DataFrame(a)
a.columns = langs
a.index = langs
a = a.abs()
a = a.round(2)
print(a.to_latex())

#Get Plot to show effect of training on transition matrix diagonals
change = {'lang': langs,'original': original,'trained':new}
changedf = pd.DataFrame(change)
ax = changedf.plot(x='original',y='trained',kind='scatter',s=50,xlim=[0,3.5],ylim=[0,3.5],figsize=(15,10),title='Effect of Training on Transition Matrix Diagonals')
def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'], point['y'], str(point['val']),size=15)
label_point(changedf.original, changedf.trained, changedf.lang, ax)
ax.set_ylabel("Original Diagonal Sum", fontsize=15)
ax.set_xlabel("Trianed Diagonal Sum", fontsize=15)
ax.title.set_size(25)
ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".3")
plt.show()

#Get hitting matrix values using Markov Chain
def gethitting(A):
    B = A[1:4,1:4]-numpy.matrix(numpy.identity(3), copy=False)
    return(-np.linalg.inv(B).dot(numpy.matrix.transpose([[1,1,1]]+A[1:4,0]))[2,0])

#Run and get Hitting Times over all languages
langs = ['en','ja','es','ar','pt','ms','ko','tr','th','fr','nl','tl','it']
kmeanshit = [0] * numpy.size(langs)
hittingtimes = [0] * numpy.size(langs)
for i in range(numpy.size(langs)):
    model = hmmtransition(langs[i],4,['posts','foll','friend'],'posts',df)
    basicmat = transitionmatrix(langs[i],4,['posts','foll','friend'],'posts',df).as_matrix()
    transmat = numpy.exp(model.dense_transition_matrix())[0:-2, 0:-2]
    kmeanshit[i] = gethitting(basicmat)
    hittingtimes[i] = gethitting(transmat)
hitting = {'Language': langs,'KMeans Time': kmeanshit,'HMM Time': hittingtimes}
hittingdf = pd.DataFrame(hitting)
print(hittingdf.set_index('Language').to_latex())






