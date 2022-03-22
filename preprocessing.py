# Preprocessing script file

def preprocessing(_df):
    '''remove last null column from raw data'''
    print(_df.columns)
    # remove last empty column
    _df = _df.iloc[:,:-1]
    # give column names
    _df.columns = ['class']+ ['feature_'+str(i) for i in range(16)]
    return _df

def extract_float(_df):
    '''extract float'''
    rows = _df.shape[0]
    cols = _df.shape[1]
    for i in range(0,rows):
        for j in range(1,cols):
            # split "feature:float" to "float"
            _df.iloc[i,j] = _df.iloc[i,j].split(':')[1]
    return _df
  
def change_objCol_to_float(_df):  
    '''convert object column to float features'''     
    _df.iloc[:,1:] = _df.iloc[:,1:].astype('float')
    return _df

def plot_unique_values(_df):
    '''plot class proportion'''
    return _df['class'].value_counts().plot.bar()

def any_missing_values(_df):
    '''check if a given dataframe has any null values'''
    print('there are null values' if _df.isnull().sum().sum()!=0 else 'no null values found')
