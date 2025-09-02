import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from operator import attrgetter
from scipy import stats

df = pd.read_csv('Salinan Online Retail Data.csv', header=0)

# Lakukan data cleansing, ubah kolom tahun dan bulan
df_clean = df.copy()

# Ubah order_date menjadi datetime
df_clean['order_date'] = df_clean['order_date'].astype('datetime64[ns]')

#  buat kolom tahun dan bulan (untuk melihat pola transaksi bulanan)
df_clean['year_month']=df_clean['order_date'].dt.to_period('M')

# menghapus semua baris tanpa customer_id
df_clean = df_clean[~df_clean['customer_id'].isna()]
# menghapus semua baris tanpa product_name
df_clean = df_clean[~df_clean['product_name'].isna()]

# membuat semua product_name berhuruf kecil
df_clean['product_name'] = df_clean['product_name'].str.lower()
# menghapus semua baris dengan product_code atau product_name test
df_clean = df_clean[(~df_clean['product_code'].str.lower().str.contains('test')) |
                    (~df_clean['product_name'].str.contains('test '))]
# membuat kolom order_status dengan nilai 'cancelled' jika order_id diawali dengan huruf 'c' dan 'delivered' jika order_id tanpa awalan huruf 'c'
df_clean['order_status'] = np.where(df_clean['order_id'].str[:1]=='C', 'cancelled', 'delivered')
# mengubah nilai quantity yang negatif menjadi positif karena nilai negatif tersebut hanya menandakan order tersebut cancelled
df_clean['quantity'] = df_clean['quantity'].abs()
# menghapus baris dengan price bernilai negatif
df_clean = df_clean[df_clean['price']>0]
# membuat nilai amount, yaitu perkalian antara quantity dan price
df_clean['amount'] = df_clean['quantity'] * df_clean['price']

# mengganti product_name dari product_code yang memiliki beberapa product_name dengan salah satu product_name-nya yang paling sering muncul
most_freq_product_name = df_clean.groupby(['product_code','product_name'], as_index=False).agg(order_cnt=('order_id','nunique')).sort_values(['product_code','order_cnt'], ascending=[True,False])
most_freq_product_name['rank'] = most_freq_product_name.groupby('product_code')['order_cnt'].rank(method='first', ascending=False)
most_freq_product_name = most_freq_product_name[most_freq_product_name['rank']==1].drop(columns=['order_cnt','rank'])
df_clean = df_clean.merge(most_freq_product_name.rename(columns={'product_name':'most_freq_product_name'}), how='left', on='product_code')
df_clean['product_name'] = df_clean['most_freq_product_name']
df_clean = df_clean.drop(columns='most_freq_product_name')

# mengkonversi customer_id menjadi string
df_clean['customer_id'] = df_clean['customer_id'].astype(str)

# menghapus outlier
df_clean = df_clean[(np.abs(stats.zscore(df_clean[['quantity','amount']]))<3).all(axis=1)]
df_clean = df_clean.reset_index(drop=True)
df_clean

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