
# coding: utf-8

# In[84]:

from pandas import DataFrame, read_csv
from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt
import pandas as pd 
import sys 
import matplotlib 
import numpy as np


get_ipython().magic(u'matplotlib inline')

country_KPI = "/home/ubuntu/workspace/country_csv.csv"
country_values = "/home/ubuntu/workspace/country_csv_values.csv"


# In[ ]:




# 
# kpi_data = pd.read_excel('C:\\Users\\PradeepB\\Machine Learning\\HCL Hackathon\\FINAL\\Customer-country_data_5-5.xlsx',header=3)
# 
# 
# kpi_data['Customer'] = kpi_data['Customer'].fillna(method='ffill')
# kpi_data['Country'] = kpi_data['Country'].fillna(method='ffill')
# kpi_data[2013] = kpi_data[2013].fillna(value=0)
# kpi_data[2014] = kpi_data[2014].fillna(value=0)
# kpi_data[2015] = kpi_data[2015].fillna(value=0)
# kpi_data[50:100]
# kpi_data = kpi_data.dropna()
# kpi = np.array(kpi_data)
# 
# 

# In[ ]:




# In[85]:


customer_country = pd.read_excel('C:\\Users\\PradeepB\\Machine Learning\\HCL Hackathon\\FINAL\\Customer-country_data_5-5.xlsx',header=9,sheetname='customer-country')
customer_country = customer_country.fillna(0)
customer_country = customer_country.replace({'\n': ''}, regex=True)
customer_country = customer_country[:730]

customer_country = customer_country.replace({',': ''}, regex=True)
customer_country = customer_country.fillna(value=0)
customer_country.dtypes
customer_country=customer_country.convert_objects(convert_numeric=True)
customer_country = customer_country.fillna(value=0)



# In[86]:

a=np.array(customer_country)
country = a[:,:4]
b=(a[:,4:].sum(axis=1))
b=np.reshape(b,(b.shape[0],1))
b=np.array(b.T)
country=np.concatenate((country, b.T), axis=1)
count=0
count=np.empty([5])
a=count
for i in country:
    count
    temp = i[0]
    i[0] = i[1]
    i[1] = temp
    count =np.vstack([count,i])
count = np.delete(count, [0], axis=0)


p = pd.DataFrame(count)
p = pd.DataFrame(count,columns=['Country','Company','KPI','Year','Value'])
k = pd.DataFrame(kpi,columns=['Country','Company','KPI','2013','2014','2015','Sum'])
p=p.sort_values(['Country','Company','KPI'], ascending=[True,True,True])
p[p.Country=='UK'][p.Company=='FORD MOTOR'][p.KPI=='Credit terms']


# In[87]:

k.sort_values(['Country','Company','KPI'], ascending=[True,True,True])
k[k.Country=='China']


# In[88]:

p[p.Country=='UK'][p.Company=='FORD MOTOR'][p.KPI=='Credit terms'].Value


# In[89]:


p = pd.DataFrame(count)
p = pd.DataFrame(count,columns=['Country','Company','KPI','Year','Value'])
k = pd.DataFrame(kpi,columns=['Country','Company','KPI','2013','2014','2015','Sum'])
p=p.sort_values(['Country','Company','KPI'], ascending=[True,True,True])
p[p.Country=='UK'][p.Company=='FORD MOTOR'][p.KPI=='Credit terms']

res = DataFrame(columns=['Va'])
p.iloc[0].Country

j=0

for i in range(p.shape[0]):
    res
    temp = p.iloc[i]
    
    a=(str(int(temp.Year)))

    res.loc[j]=(k[k.Country==temp.Country][k.Company==temp.Company][k.KPI==temp.KPI][a]*temp.Value).iloc[0]
   

    j=j+1
res[430:]
p=p.reset_index(drop=True)
p=p.join(res)



# In[90]:


k[k.Country=='China'][k.Company=='PEPSICO'][k.KPI=='Credit terms']['2014']


# In[91]:

p[p.Country=='UK'][p.Company=='FORD MOTOR'][p.KPI=='Credit terms']


# In[92]:

p[209:]


# In[93]:

temp=pd.get_dummies(p,columns=['KPI','Year'])


# In[94]:

del temp['Value']


# In[95]:

temp[210:]


# In[96]:

data = kpi_data = pd.read_excel('C:\\Users\\PradeepB\\Machine Learning\\HCL Hackathon\\FINAL\\Opportunities_examples.xlsx')
data.dtypes


# In[97]:

data=data.convert_objects(convert_numeric=True)
data.dtypes


# In[98]:

new_X = data.filter(['Opp Type','Status','Region','Country','BU','Sector','Customer Group','Customer','Product','Payment Terms','Contract Term','Milestone','Ann Rev (€m)','Ann GP (€m)'], axis=1)
temp = temp.rename(columns={'Company': 'Customer Group'})
new_y = data.filter(['Prob to win %'])


# In[99]:

new_X


# In[100]:


temp2 = pd.get_dummies(new_X,columns=['Opp Type','Status','Region','BU','Sector','Customer','Product','Contract Term','Milestone'])
temp2


temp2.set_index(['Customer Group','Country'])

temp.set_index(['Customer Group','Country'])
temp3 = pd.merge(temp2, temp,  how='left', left_on=['Customer Group','Country'], right_on = ['Customer Group','Country'])
temp2


# In[ ]:




# In[101]:

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


# In[102]:

temp


# In[103]:


arr = ["KPI_Credit terms","KPI_DSO","KPI_Gross Profit",
    "KPI_Inventory Location Accuracy",
    "KPI_Lost Time Injury Frequency Rate",
    "KPI_On Time Delivery",
    "KPI_On Time Shipping","KPI_Shipping Accuracy",
    "KPI_Third party trade receivables",
    "KPI_Total Net Revenue"]
arr2 = ["Country","Customer Group","Va","KPI_Credit terms","KPI_DSO","KPI_Gross Profit",
    "KPI_Inventory Location Accuracy",
    "KPI_Lost Time Injury Frequency Rate",
    "KPI_On Time Delivery",
    "KPI_On Time Shipping","KPI_Shipping Accuracy",
    "KPI_Third party trade receivables",
    "KPI_Total Net Revenue"]
j=0
df = pd.DataFrame(columns=arr2)

z1=0
z2=0

while z1<temp.shape[0]:
    z2=z1
    while z1<temp.shape[0] and temp.iloc[z1][arr[j]]==1:
        z1=z1+1
    

    a=(pd.DataFrame([temp.iloc[z2]['Country']],columns=['Country']).sum())
    b=(pd.DataFrame([temp.iloc[z2]['Customer Group']],columns=['Customer Group']).sum())
    c=(temp[z2:z1].sum(numeric_only=True))
    result = pd.concat([a,b,c], axis=0)
    
    result[arr[j]] = result['Va']
    j=(j+1)%10
    del result['Year_2013']
    del result['Year_2014']
    del result['Year_2015']

    df=df.append(result, ignore_index=True)

df1=df


# In[104]:

df = pd.DataFrame(columns=arr2)
for i in range(df1.shape[0]/10):

    a=(pd.DataFrame([df1.iloc[i*10]['Country']],columns=['Country']).sum())
    b=(pd.DataFrame([df1.iloc[i*10]['Customer Group']],columns=['Customer Group']).sum())
    c=(df1[i*10:(i+1)*10].sum(numeric_only=True))
    result = pd.concat([a,b,c], axis=0)
    df=df.append(result,ignore_index=True)

    del df['Va']
df


# In[105]:

def get_yearvar(year):
    import requests
    import numpy as np
    total = list()
    url = 'https://www.quandl.com/api/v3/datasets/SSE/DPW.json?start_date='+str(year)+'-01-01&end_date='+str(year+1)+'-01-01'
    response = requests.get(url)
    if response.status_code != 200:
        return None
    data = response.json()
    for item in data['dataset']['data']:
        if item[0].find(str(year)) == -1:
            continue;
        total.append(item[1])
    return np.var(total)


# In[106]:

data = kpi_data = pd.read_excel('C:\\Users\\PradeepB\\Machine Learning\\HCL Hackathon\\FINAL\\Opportunities_examples.xlsx')
data=data.replace(to_replace='HEWLETT PACKARD ENTERPRISE',value='HEWLETT PACKARD')
data=data.convert_objects()
data.dtypes
new_X = data.filter(['Opp Type','Status','Region','Country','Sector','Customer Group','Product','Contract Term','Payment Terms','Milestone','Ann Rev (€m)','Ann GP (€m)'], axis=1)
temp = temp.rename(columns={'Company': 'Customer Group'})
new_y = data.filter(['Prob to win %'])

new_X = pd.merge(new_X, df,  how='left', left_on=['Customer Group','Country'], right_on = ['Customer Group','Country'])

new_X=pd.get_dummies(new_X,columns=['Opp Type','Status','Region','Sector','Product','Country','Customer Group','Milestone','Contract Term'])
new_X=new_X.fillna(0)
new_y=new_y.fillna(0)
new_X['Stock Variance'] = pd.Series(np.full(65, fill_value = get_yearvar(2015)))


# In[ ]:




# In[175]:

col=[
    
    "KPI_Credit terms",

"KPI_Gross Profit",
"KPI_Inventory Location Accuracy",
"KPI_Lost Time Injury Frequency Rate",
"KPI_On Time Delivery",
"KPI_On Time Shipping",

"KPI_Third party trade receivables",
"KPI_Total Net Revenue",
    
"Milestone_Closed - Canceled",
"Milestone_Closed - Lost",
"Milestone_Contract Signed",
"Milestone_Customer commitment– Move to gain",
"Milestone_Data quality assessment conducted",
"Milestone_Early Lead",
"Milestone_Potential Opportunity",
"Milestone_Proposal (incl. COO & COS) signed off",
"Milestone_Proposal and solution fit presented",
"Milestone_Qualified and signed off by Sponsor",
"Milestone_Shortlisted",
"Milestone_Verbal Customer Commitment Received",
    

    
"Contract Term_1-2 Years",
"Contract Term_2-3 Years",
"Contract Term_3-5 Years",
"Contract Term_7+ Years",
"Contract Term_< 1 Year"
    
    ]


# In[108]:

def myround(x, base=5):
    return int(base * round(float(x)/base))



# In[183]:

new_X=new_X.fillna(0)

new_y =new_y.fillna(0)
te=new_X.as_matrix(columns=col)
new = new_X.filter(col, axis=1)
from sklearn.cross_validation import train_test_split
from sklearn import datasets, linear_model
X_train, X_test, y_train, y_test = train_test_split(new, new_y, random_state=1999)
regr=linear_model.LinearRegression()
R1=regr
regr.fit(X_train,y_train)


print(regr.score(X_test,y_test))
l1_pred=regr.predict(X_test)
l11_pred=regr.predict(X_train)
yy_train = y_train
yy_test = y_test


# In[219]:

print(X_train.shape)


# In[186]:

print(X_test)


# from sklearn.ensemble import RandomForestRegressor
# regrrr = RandomForestRegressor(max_depth=4, random_state=10)
# regrrr.fit(X_train,y_train)
# regrrr.score(X_test,y_test)

# In[110]:

# Dark colours indicate lower density 
# Bright colours indicate higher density
for key in new_X.keys():
    x = new_X[key]
    y = new_y['Prob to win %']
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)
    fig, ax = plt.subplots()
    ax.scatter(x, y, c=z, s=100, edgecolor='')
    plt.ylabel('Prob to win %')
    plt.xlabel(key)
    plt.show()
    plt.close()


# In[111]:

regr.score(X_test,y_test)


# In[112]:

col=[
    

"Status_1. Gain",
"Status_2. Opportunity",
"Status_3. Potential Opps",
"Status_4. Early Lead",
"Status_5. Lost",
"Status_7. Cancelled"
    

    
    ]
new_X=new_X.fillna(0)

new_y =new_y.fillna(0)
te=new_X.as_matrix(columns=col)
new = new_X.filter(col, axis=1)
X_train, X_test, y_train, y_test = train_test_split(new, new_y, random_state=1999)


# In[ ]:




# In[168]:

r= linear_model.LogisticRegression()
R2=r
r.fit(X_train,y_train)
r.score(X_test,y_test)


# In[114]:

l22_pred=r.predict(X_train)


# In[ ]:




# In[115]:

l11_pred.shape


# In[116]:

l22_pred.shape


# In[ ]:




# In[ ]:




# In[169]:

l2_pred=r.predict(X_test)
l22_pred=r.predict(X_train)
l2_pred=l2_pred.reshape(17,1)
l22_pred=l22_pred.reshape(48,1)
con=np.concatenate((l11_pred,l22_pred),axis=1)
fi_X_train, fi_X_test, fi_y_train, fi_y_test = train_test_split(con, yy_train, random_state=2)
regr=linear_model.LinearRegression()
R3=regr
regr.fit(fi_X_train,fi_y_train)

print ("The final score for Probability to win % model is",regr.score(fi_X_test,fi_y_test)*100,"%")


# In[118]:

churnArr=[]

def fun(x):
    for i in x:
        if i[0]<0:
            print(0)
            churnArr.append(100)
        elif i[0]>100:
            print(100)
            churnArr.append(0)
        else:
            print (int(5 * round(float(i[0])/5)))
            churnArr.append(int(5 * round(float(i[0])/5)))


# In[119]:

fun(regr.predict(fi_X_test))


# In[120]:

fi_y_test


# In[121]:

print ("Churn Rate of the Company is: ",np.mean(churnArr))


# In[284]:

def fetFun(KPI_Credit_terms,
KPI_Gross_Profit,
KPI_Inventory_Location_Accuracy,
KPI_Lost_Time_Injury_Frequency_Rate,
KPI_On_Time_Delivery,
KPI_On_Time_Shipping,
KPI_Third_party_trade_receivables,
KPI_Total_Net_Revenue,
Milestone,Contract_terms,Status):
    KPI=["KPI_Credit terms",

    "KPI_Gross Profit",
    "KPI_Inventory Location Accuracy",
    "KPI_Lost Time Injury Frequency Rate",
    "KPI_On Time Delivery",
    "KPI_On Time Shipping",

    "KPI_Third party trade receivables",
    "KPI_Total Net Revenue"]
    
    Mile = ["Closed - Canceled",
    "Closed - Lost",
    "Contract Signed",
    "commitment– Move to gain",
    "Data quality assessment conducted",
    "Early Lead",
    "Potential Opportunity",
    "Proposal (incl. COO & COS) signed off",
    "Proposal and solution fit presented",
    "Qualified and signed off by Sponsor",
    "Shortlisted",
    "Verbal Customer Commitment Received"]
    
    Stat = ["1. Gain",
    "2. Opportunity",
    "3. Potential Opps",
    "4. Early Lead",
    "5. Lost",
    "7. Cancelled"]
    
    contTe=["1-2 Years",
    "2-3 Years",
    "3-5 Years",
    "7+ Years",
    "1 Year"]
    
    arr1=[]
    arr1.append(KPI_Credit_terms)
    arr1.append(KPI_Gross_Profit)
    arr1.append(KPI_Inventory_Location_Accuracy)
    arr1.append(KPI_Lost_Time_Injury_Frequency_Rate)
    arr1.append(KPI_On_Time_Delivery)
    arr1.append(KPI_On_Time_Shipping)
    arr1.append(KPI_Third_party_trade_receivables)
    arr1.append(KPI_Total_Net_Revenue)
    for i in Mile:
        if i == Milestone:
            arr1.append(1)
        else:
            arr1.append(0)
    for i in contTe:
        if i == Contract_terms:
            arr1.append(1)
        else:
            arr1.append(0)

    
    arr2 = []
    for i in Stat:
        if i == Status:
            arr2.append(1)
        else:
            arr2.append(0)
    
#     print((pd.DataFrame(data=arr1,index=col).shape))
    del arr1[-1]
#     print(len(arr1))
#     print(arr1)
    a=(R1.predict(arr1))
    b=(R2.predict(arr2))
    print(R3.predict([a[0],b[0]]))
    


# In[285]:

fetFun(984.31,-1.5e+06,3387.0935,0.027290,3218.1538,3400.00,3388.457,5.275e+07,"Closed - Lost","< 1 Year","5. Lost")


# In[ ]:




# In[ ]:




# In[ ]:



