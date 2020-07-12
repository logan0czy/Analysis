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
df_data.reset_index(...)

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
## Summary & Maps  
```python
s.dtype | df.dtypes | s.astype('...') | df.shape
# different dtypes have different summaries
df.describe()  # S.describe()

# other statistical description
df.col.mean()
df.col.unique()
df.col.value_counts()
```
**Map Function**对数据进行逐项的变化，但是如果有pandas原生的函数的话用原生函数底层速度优化更好  
```python
# map
S.map(lambda x: ...)
# apply
df.apply(func, axis='columns/index')
```
## Grouping  
将DataFrame按一项或者几项数据进行`顺序`分集，grouping的结果虽然不直接可见，但是能够在此基础上对各个分集了的数据进行分别处理。  
```python
df.groupby('col_name')[col].statistical_func()  # 'col' can be the selected group col or other cols
df.groupby(['col_1', 'col_2'])
*****agg function*****
df.groupby('col_name')[col].agg([func1, func2, func3])

agg_df = df.groupby('col_name').agg({'col_1': [func1, func2], 'col_2': [func3, func4]})

# 将列层级重新命名后展开
agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
```
多项特征的group会使得DataFrame具有`Multi-Index`，可以调用`df.reset_index()`方法来将层级进行展开
**Sorting**
```python
# by column
df.sort_values(by='col_name', ascending=False)
df.sort_values(by=['col_1', 'col_2'])  # sort by multiple column
# by index
df.sort_index()
```
## Missing Values  
基础填充操作`S.fillna(...)`；基础置换操作`S.replace(raw, alternative)`  
## Renaming & Combining  
**Renaming**  
```python
df.rename(index={'index_origin': 'index_new'},
          columns={'col_origin': 'col_new', ...})
*****rename row index and column index*****
reviews.rename_axis("wines", axis='rows').rename_axis("fields", axis='columns')
```
**Combining**`.concat(), .join(), .merge()`, mostly the first two  
```python
# join combine differenct DataFrame objects which have an index in common
left.join(right, lsuffix='_CAN', rsuffix='_UK')
```
