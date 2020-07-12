# Pandas常用操作整理  
Pandas是数据分析中常用的库，相比于底层的numpy，可以极大简化数据的操作和分析的效率，将精力更多地用于分析数据特征而不是在怎样进行编写数据处理的代码中  
```python
import pandas as pd
# I/O
pd.read_csv(fpath, index_col=int/col_name, parse_date=[...])
df.to_csv(fpath, index=False)

# Series
pd.Series([...], index=[...], name='...')
# DataFrame
pd.DataFrame({'col_1': [...], 'col_2': [...]}, index=[...])
```
## Indexing, Selecting & Assigning  
```python
# basics
df.country or df['country']
# Manipulate Index
df_data.set_index('col_name')

*****Index-based*****
# same with python, last element excluded
df_data.iloc[row int/slice/list, column int/slice/list]

*****Label-based*****
# last element included
df_data.loc[index name/name slice/list, column name/name list]

*****Conditional Selection*****
reviews.loc[reviews.country == 'Italy']
reviews.loc[(reviews.country == 'Italy') & (reviews.points >= 90)]  # or '|'

reviews.loc[reviews.country.isin(['Italy', 'France'])]
reviews.loc[reviews.price.notnull()]  # or isnull
```
