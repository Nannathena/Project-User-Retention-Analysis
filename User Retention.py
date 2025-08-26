import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from operator import attrgetter


df = pd.read_csv('Salinan Online Retail Data.csv', header=0)

# Lakukan data cleansing, ubah kolom tahun dan bulan
df_clean = df.copy()

# Ubah order_date menjadi datetime
df_clean['order_date'] = pd.to_datetime(df_clean['order_date'], errors='coerce') 

#  buat kolom tahun dan bulan (untuk melihat pola transaksi bulanan)
df_clean['year_month']=df_clean['order_date'].dt.to_period('M')

# Agregat data transaksi ke bentuk summary total transaksi/order setiap pengguna setiap bulan
df_user_monthly = df_clean.groupby(['customer_id','year_month'], as_index=False).agg(order_count=('order_id','unique'))

# Buat kolom sebagai cohort dari pengguna
df_user_monthly['cohort'] = df_user_monthly.groupby('customer_id')['year_month'].transform('min')

df_user_monthly['period_num']=(df_user_monthly['year_month']-df_user_monthly['cohort']).apply(attrgetter('n'))+1

df_cohort_pivot= pd.pivot_table(df_user_monthly,index='cohort',columns='period_num',values='customer_id', aggfunc=pd.Series.nunique)

# Buat cohort menjadi persentase
cohort_size = df_cohort_pivot.iloc[:,0]
df_retention_cohort = df_cohort_pivot.divide(cohort_size,axis=0)

with sns.axes_style('white'):
    fig,ax= plt.subplots(1,2,figsize=(12,8),sharey= True, gridspec_kw={'width_ratios':[1,11]})
    
    sns.heatmap(df_retention_cohort,annot=True, fmt='.0%', cmap='RdYlGn', ax=ax[1])
    ax[1].set_title('User Retention Cohort')
    ax[1].set(xlabel='Month Number', ylabel='')
    
    df_cohort_size=pd.DataFrame(cohort_size)
    white_cmap=mcolors.ListedColormap(['white'])
    sns.heatmap(df_cohort_size,annot=True,cbar=False,fmt='g',cmap=white_cmap,ax=ax[0])
    ax[0].tick_params(bottom=False)
    ax[0].set_yticklabels(ax[0].get_yticklabels(), fontsize=9)  
    ax[0].set(xlabel='Cohort Size',ylabel='First Order Month',xticklabels=[])
    
    fig.tight_layout()
    plt.show()
    


# print(df_retention_cohort.head())
print(df_cohort_size.head())