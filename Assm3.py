import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import scipy.optimize as opt


#%% Clustering

df_data = pd.read_excel('API_19_DS2_en_excel_v2_21245.xls') #Importing data

df_new = df_data.iloc[2:]

filtered_df = df_new[df_new['Unnamed: 2'].isin(['Urban population','Population, total','Renewable electricity output (% of total electricity output)','Renewable energy consumption (% of total final energy consumption)'])] #Creating a new df with the filtered results

filtered_df['Row'] = filtered_df.groupby('Data Source').cumcount()



pivoted = filtered_df.pivot(index=['Data Source'], columns='Unnamed: 2', values='Unnamed: 4')


pivoted = pivoted.reset_index()

pivoted = pivoted.dropna()

pivoted[['Urban population','Population, total']] = pivoted[['Urban population','Population, total']].astype(int)  #Pivotes my data frame so that the coulmns become the headings

cols = pivoted.columns[1:] 


pivoted[cols] = pivoted[cols].apply(pd.to_numeric)

corr = pivoted.corr(numeric_only=True)


plt.figure()
plt.imshow(corr)  #The colour bar graph
plt.colorbar()

plt.xticks(ticks = [0, 1, 2, 3], rotation=30,labels = ['Pop','Renewable Output','Renewable Consumption','Urban Pop'])
plt.yticks([0, 1, 2, 3], labels = ['Pop','Renewable Output','Renewable Consumption','Urban Pop'], rotation=0)
plt.show()

pd.plotting.scatter_matrix(pivoted, figsize=(10, 10), s=10)
plt.show()


plt.figure()
plt.scatter( pivoted["Population, total"],pivoted["Renewable electricity output (% of total electricity output)"], 10, marker="o") #First clusters
plt.xlabel("Population Total")
plt.ylabel("Renwable energy: % of total Output")
plt.show()

scaler = pp.RobustScaler()


df_clust = pivoted[["Population, total","Renewable electricity output (% of total electricity output)"]]
# and set up the scaler
scaler.fit(df_clust)

df_norm = scaler.transform(df_clust)


plt.figure(figsize=(8, 8))
plt.scatter(df_norm[:,0], df_norm[:, 1], 10, marker="o")
plt.xlabel("Population")
plt.ylabel("Renewable energy Output")
plt.show()


def one_silhoutte(xy, n):  #calculates the silhoutte scores


    kmeans = cluster.KMeans(n_clusters=n, n_init=20)

    kmeans.fit(xy) 

    labels = kmeans.labels_

    score = (skmet.silhouette_score(xy, labels))
    
    return score

for ic in range(2, 11):
    score = one_silhoutte(df_norm, ic)
    print(f"The silhouette score for {ic: 3d} is {score: 7.4f}")
    
kmeans = cluster.KMeans(n_clusters=2, n_init=20)
# Fit the data, results are stored in the kmeans object
kmeans.fit(df_norm) # fit done on x,y pairs
# extract cluster labels
labels = kmeans.labels_

cen = kmeans.cluster_centers_
cen = scaler.inverse_transform(cen)
xkmeans = cen[:, 0]
ykmeans = cen[:, 1]

x = df_clust["Population, total"]

y = df_clust["Renewable electricity output (% of total electricity output)"]

plt.figure(figsize=(8.0, 8.0))
# plot data with kmeans cluster number
plt.scatter(x, y, 10, labels, marker="o", cmap = 'bwr')
# show cluster centres
plt.scatter(xkmeans, ykmeans, 45, "black", marker="d")

plt.xlabel("Population")
plt.ylabel("Renewable electricity output")
plt.show()

#%% Fitting

df_fdata = pd.read_excel('fitting.xls')

df_fdata = df_fdata[2:]

mask = df_fdata['Data Source'] == 'World'
mask.iloc[0] = True  # Force include the first row

df_fdata = df_fdata[mask].reset_index(drop=True)

mask = df_fdata['Unnamed: 2'] == 'Population, total'

mask.iloc[0] = True  # Force include the first row

df_fdata = df_fdata[mask].reset_index(drop=True)

df_fdata = df_fdata.drop(df_fdata.columns[:4], axis=1)

df_fdata = df_fdata.drop("Unnamed: 68", axis = 1)

df_fdata = df_fdata.astype(int)

print(df_fdata.dtypes)



print(df_fdata.iloc[0])

plt.figure()

plt.plot(df_fdata.iloc[0],df_fdata.iloc[1])

plt.ticklabel_format(style='plain', axis='both')


plt.show()

def exponential(t, n0, g):

    t = t - 1990
    f = n0 * np.exp(g*t)

    return f

param, covar = opt.curve_fit(exponential, df_fdata.iloc[0], df_fdata.iloc[1])

plt.figure()
plt.plot(df_fdata.iloc[0], exponential(df_fdata.iloc[0], 5.2e9, 0.02))
plt.plot(df_fdata.iloc[0], df_fdata.iloc[1])
plt.ticklabel_format(style='plain', axis='both')
plt.xlabel("Year")
plt.legend()
plt.show()

param, covar = opt.curve_fit(exponential, df_fdata.iloc[0], df_fdata.iloc[1], p0=(5.2e9, 0.02))
print(f"GDP 1990: {param[0]/1e9:6.1f} billion $")
print(f"growth rate: {param[1]*100:4.2f}%")

plt.figure()
plt.plot(df_fdata.iloc[0], exponential(df_fdata.iloc[0], 5.1e9, 0.015),label = 'Fit')
plt.plot(df_fdata.iloc[0], df_fdata.iloc[1],label = 'Population')
plt.ticklabel_format(style='plain', axis='both')
plt.xlabel("Year")
plt.legend()
plt.show()








