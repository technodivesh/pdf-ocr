# resize

scale_percent = 50 # percent of original size
width = int(self.img_bin.shape[1] * scale_percent / 100)
height = int(self.img_bin.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
resized = cv.resize(self.img_bin, dim, interpolation = cv.INTER_AREA)
resized_col = cv.resize(self.img, dim, interpolation = cv.INTER_AREA)



df.replace('',np.nan,inplace=True)
df.dropna(axis=0, how='all', thresh=None, subset=None, inplace=True)
df.replace(np.nan,'',inplace=True)

df[df.columns[-1]].replace('',np.nan,inplace=True)
df[df.columns[-1]].fillna(method='bfill',inplace = True)
dfs = df.groupby(df.columns[-1],sort=False)


dfs = list(_df.drop(_df.index[-1]) for i,_df in dfs if len(_df)>1)

for df in dfs:
    DROP = []
    CARRIER = ''
    
    for i,row in enumerate(list(df.iterrows())): 
        if df.iloc[i][1:-1].isna().all():
            
            CARRIER = df.iloc[i][0] 
            DROP.append(i)

    if DROP:
        df.drop(df.index[DROP[0]],inplace=True)


print(dfs)
frames = [df for df in dfs]
result = pd.concat(frames)