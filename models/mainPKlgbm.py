from data.load import load_data
import pandas as pd
from random import sample
import numpy as np
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor

import lightgbm as lgbm
#%%
random_n=100000 #0 takes all of them
price_strategy='Log'
competitor_strategy='Keep'
orig_destination_distance_strategy='Drop'
visitors_strategy='KNN'
hotel_data_strategy='Keep'
month_strategy='Extract_Encode'
year_strategy='Extract_Encode'
diff_features_strategy='Yes'
#normalization=(['prop_review_score','prop_location_score1','price_usd'],['srch_id','prop_id','srch_booking_window','srch_destination_id','prop_country_id'])
normalization=(['prop_review_score','prop_location_score1','price_usd','prop_location_score_3'],['srch_id','prop_id','srch_booking_window','srch_destination_id','prop_country_id'])

plots=False
#columns_to_drop=[]
columns_to_drop=['prop_id']
test_or_cv='Test'
#%% Basic plotting for EDA
if plots:
	df = load_data(caching=False)
	corr=df.drop(columns=['srch_id','visitor_location_country_id','date_time','site_id','prop_country_id','prop_id','srch_destination_id','gross_bookings_usd']).corr()
	
	# Set up the matplotlib figure
	f, ax = plt.subplots(figsize=(16, 12))
	# Generate a custom diverging colormap
	cmap = sns.diverging_palette(220, 20, as_cmap=True)
	# Draw the heatmap with the mask and correct aspect ratio
	sns.heatmap(corr, cmap=cmap, vmax=1, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .6})
	
	df.hist(bins=30, figsize=(15, 10))
	
	#msno.matrix(df)
	df.isnull().sum().sort_values(ascending=False)
	
	def bar_plots(column):
		df_temp1=df.loc[df['booking_bool']==1,column].value_counts(dropna=False).divide(df.loc[:,column].value_counts(dropna=False))
		df_temp2=df.loc[df['click_bool']==1,column].value_counts(dropna=False).divide(df.loc[:,column].value_counts(dropna=False))
		df_temp=pd.merge(df_temp1,df_temp2,how='outer',left_index=True,right_index=True,suffixes=['_booking','_click'])
		df_temp.plot.bar()	
		
		
	for col in ['site_id', 'visitor_location_country_id','prop_country_id','prop_brand_bool',
	 'position', 'promotion_flag','srch_length_of_stay', 'srch_booking_window','srch_adults_count', 'srch_children_count', 'srch_room_count',
	 'random_bool', 'comp1_rate', 'comp1_inv','comp2_rate', 'comp2_inv', 'comp3_rate', 'comp3_inv', 'comp4_rate', 'comp4_inv',
	'comp5_rate', 'comp5_inv','comp6_rate', 'comp6_inv','comp7_rate', 'comp7_inv','comp8_rate', 'comp8_inv', 'click_bool']:
		bar_plots(col)
	
	
	for col in df.columns:
		sns.kdeplot(data=df[[col,'booking_bool']],x=col,common_norm=False,hue='booking_bool')
		plt.show()
		
		sns.kdeplot(data=df[[col,'click_bool']],x=col,common_norm=False,hue='click_bool')
		plt.show()
#%% Dataset Transformation
#Part I Merging
df = load_data()
df['Train']=1
df_test=load_data(caching=False,test=True)
df_test['Train']=0
df_test['srch_id']=df_test['srch_id']+1000000
df_all=pd.concat([df,df_test],axis=0)

del df, df_test
#df_all.to_pickle('df_all.pkl')
#%% Part II proper transformations
df_all=pd.read_pickle('df_all.pkl')
comp_columns=['comp1_rate', 'comp1_inv',
      'comp1_rate_percent_diff', 'comp2_rate', 'comp2_inv',
      'comp2_rate_percent_diff', 'comp3_rate', 'comp3_inv',
      'comp3_rate_percent_diff', 'comp4_rate', 'comp4_inv',
      'comp4_rate_percent_diff', 'comp5_rate', 'comp5_inv',
      'comp5_rate_percent_diff', 'comp6_rate', 'comp6_inv',
      'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv',
      'comp7_rate_percent_diff', 'comp8_rate', 'comp8_inv',
      'comp8_rate_percent_diff']

#Competitor strategy
if competitor_strategy=='Drop':
	df_trimmed1=df_all.drop(columns=comp_columns)
else:
	df_trimmed1=df_all.copy()
	df_trimmed1.loc[:,comp_columns]=df_trimmed1.loc[:,comp_columns].fillna(0)

del df_all

#Price Startegy
if price_strategy=='Log':
	df_trimmed1['price_usd']=np.log(1+df_trimmed1['price_usd'])
else:
	pass
# Origin Destination Strategy
if orig_destination_distance_strategy=='Drop':
	df_trimmed1.drop(columns='orig_destination_distance',inplace=True)

# Visitors strategy	
if visitors_strategy=='Drop':
	df_trimmed1.drop(columns=['visitor_hist_starrating','visitor_hist_adr_usd'],inplace=True)
elif visitors_strategy=='Keep':
	df_trimmed1[['visitor_hist_starrating','visitor_hist_adr_usd']].fillna(0,inplace=True)
elif visitors_strategy=='KNN':

	regressors=['srch_id','site_id', 'visitor_location_country_id','prop_country_id',
	       'srch_destination_id', 'srch_length_of_stay', 'srch_booking_window',
	       'srch_adults_count', 'srch_children_count', 'srch_room_count']
	var1=['visitor_hist_adr_usd']
	var2=['visitor_hist_starrating']
	df_kn=df_trimmed1[regressors+var1+var2].drop_duplicates()

	df_kn.isnull().sum().sort_values(ascending=False)

	df_train_au=df_kn.loc[df_kn['visitor_hist_adr_usd']>=-10000].drop(columns=var2)
	df_test_au=df_kn.loc[df_kn['visitor_hist_adr_usd'].isna()].drop(columns=var2+var1)
	param_grid = { 'n_neighbors': [5, 10,25]}
	neigh_au = KNeighborsRegressor()
	grid_neigh_au = GridSearchCV(neigh_au, param_grid, cv=5,verbose=1)
	grid_neigh_au.fit(df_train_au.drop(columns=var1),df_train_au[var1])
	neigh_au_best=grid_neigh_au. best_estimator_
	pred_au=neigh_au_best.predict(df_test_au)
	df_pred_au=pd.DataFrame(index=df_test_au['srch_id'],data=pred_au,columns=['adr_usd'])

	df_train_sr=df_kn.loc[df_kn['visitor_hist_starrating']>=-100000].drop(columns=var1)
	df_test_sr=df_kn.loc[df_kn['visitor_hist_starrating'].isna()].drop(columns=var1+var2)
	neigh_sr = KNeighborsRegressor()
	grid_neigh_sr = GridSearchCV(neigh_sr, param_grid, cv=5,verbose=1)
	grid_neigh_sr.fit(df_train_sr.drop(columns=var2),df_train_sr[var2])
	neigh_sr_best=grid_neigh_sr. best_estimator_
	pred_sr=neigh_sr_best.predict(df_test_sr)
	df_pred_sr=pd.DataFrame(index=df_test_sr['srch_id'],data=pred_sr,columns=['star'])

	df_big=pd.merge(df_pred_au,df_pred_sr,how='outer',left_index=True,right_index=True).reset_index()
	df_trimmed1=pd.merge(df_trimmed1,df_big,how='left',on='srch_id')

	df_trimmed1.loc[df_trimmed1['visitor_hist_adr_usd'].isna(),'visitor_hist_adr_usd']=df_trimmed1.loc[df_trimmed1['visitor_hist_adr_usd'].isna(),'adr_usd']
	df_trimmed1.loc[df_trimmed1['visitor_hist_starrating'].isna(),'visitor_hist_starrating']=df_trimmed1.loc[df_trimmed1['visitor_hist_starrating'].isna(),'star']
	df_trimmed1.drop(columns=['star','adr_usd'],inplace=True)

	del df_big,df_kn,df_pred_au,df_pred_sr,df_train_au,df_train_sr,df_test_au,df_test_sr

#msno.matrix(df_trimmed1)
df_trimmed1.isnull().sum().sort_values(ascending=False)

#Hotel Data strategy
if hotel_data_strategy=='Drop':
	df_trimmed2=df_trimmed1.drop(columns=['srch_query_affinity_score','prop_location_score2'])
else:
	df_trimmed2=df_trimmed1.copy()
	df_trimmed2['srch_query_affinity_score'].fillna(df_trimmed2['srch_query_affinity_score'].min(),inplace=True)
	df_trimmed2.groupby('prop_id')[['booking_bool']].agg(np.nanmean)
	df_trimmed2.groupby('prop_id')[['click_bool']].agg(np.nanmean)
	temp=pd.merge(df_trimmed2.groupby('prop_id')[['booking_bool']].agg(np.nanmean),df_trimmed2.groupby('prop_id')[['click_bool']].agg(np.nanmean),left_index=True,right_index=True,how='outer').fillna(0)
	temp['prop_location_score_3']=temp['booking_bool']*5+temp['click_bool']
	df_trimmed2=pd.merge(df_trimmed2,temp['prop_location_score_3'].reset_index(),how='left',on='prop_id')
	df_trimmed2.drop(columns='prop_location_score2',inplace=True)
del df_trimmed1
df_trimmed2.isnull().sum().sort_values(ascending=False)

df_trimmed2['prop_review_score'].fillna(0,inplace=True)

#Month strategy
if month_strategy=='Extract':
	df_trimmed2['Month']=pd.DatetimeIndex(df_trimmed2['date_time']).month
elif month_strategy=='Extract_Encode':
	df_trimmed2['Month']=pd.DatetimeIndex(df_trimmed2['date_time']).month
	df_trimmed2=pd.get_dummies(df_trimmed2, columns=["Month"])

#Year strategy
if month_strategy=='Extract':
	df_trimmed2['Year']=pd.DatetimeIndex(df_trimmed2['date_time']).year
elif month_strategy=='Extract_Encode':
	df_trimmed2['Year']=pd.DatetimeIndex(df_trimmed2['date_time']).year
	df_trimmed2=pd.get_dummies(df_trimmed2, columns=["Year"])

if diff_features_strategy=='Yes':
	df_trimmed2['usd_diff'] = np.abs(df_trimmed2.visitor_hist_adr_usd.values - df_trimmed2.price_usd.values)
	df_trimmed2['star_diff'] = np.abs(df_trimmed2.visitor_hist_starrating.values - df_trimmed2.prop_starrating.values)

#Normalization
if normalization=='None':
	pass
else:
	a,b=normalization
	for col_a in a:
		for col_b in b:
			col_name=col_a+'_norm_' +col_b

			df_trimmed2=pd.merge(df_trimmed2,df_trimmed2.groupby(col_b)[col_a].agg([np.nanmean,np.nanstd]).reset_index(),on=col_b,how='left')
			df_trimmed2['nanstd'].fillna(0,inplace=True)
			df_trimmed2.loc[df_trimmed2['nanstd']==0,'nanstd']=1
			df_trimmed2.loc[df_trimmed2['nanstd']==0,'nanmean']=0
			df_trimmed2[col_name]=df_trimmed2[col_a].subtract(df_trimmed2['nanmean'])
			df_trimmed2[col_name]=df_trimmed2[col_name].divide(df_trimmed2['nanstd'])
			df_trimmed2.drop(columns=['nanmean','nanstd'],inplace=True)
		
			print(col_name)
			
		df_trimmed2.drop(columns=col_a,inplace=True)

### feature which highlights difference between the visitors historical data and the hotel cost
		
# Duplicated rows 
df_trimmed2.drop_duplicates(subset=['srch_id','prop_id','position'],inplace=True)
#df_trimmed2.to_pickle('df_trimmed2_ndr.pkl')
#%% Part III Train set
#df_trimmed2=pd.read_pickle('df_trimmed2_ndr.pkl')

df=df_trimmed2
del df_trimmed2
#(df.groupby('srch_id')['srch_id'].count()).index[(df.groupby('srch_id')['srch_id'].count()>40).values]
df['Target']=df['click_bool'].copy()
df.loc[df['booking_bool']==1,'Target']=df.loc[df['booking_bool']==1,'booking_bool']
not_in_train=['gross_bookings_usd','click_bool','booking_bool','position']

df_train_whole=df.loc[df['Train']==1,:].sort_values(['srch_id','position'],ascending=[True,True]).drop(columns=not_in_train)
df_test=df.loc[df['Train']==0,:].drop(columns=not_in_train+['date_time','Train'])
#%%
if test_or_cv=='Test':
	if random_n>0:
		srch_ids=sample(list(df_train_whole['srch_id'].unique()),random_n)
	else:
		srch_ids=list(df_train_whole['srch_id'].unique())
		
	
	df_train=df_train_whole.loc[df_train_whole['srch_id'].isin(srch_ids),:].drop(columns=['date_time','Train']+columns_to_drop)
	df_eval=df_train_whole.loc[~df_train_whole['srch_id'].isin(srch_ids),:].drop(columns=['date_time','Train']+columns_to_drop)
	del df_train_whole
	
	qids_train = df_train.groupby('srch_id')['srch_id'].count().to_numpy()
	X_train=df_train.drop(columns=['Target','srch_id'])
	y_train=df_train['Target']
	
	qids_eval = df_eval.groupby('srch_id')['srch_id'].count().to_numpy()
	X_eval=df_eval.drop(columns=['Target','srch_id'])
	y_eval=df_eval['Target']
	
	model = lgbm.LGBMRanker(objective='lambdarank', metric='ndcg', boosting='dart',
	                        num_leaves=255, min_data_in_leaf=1, min_sum_hessian_in_leaf=100, n_estimators=1000,
	                        num_iterations=1000, learning_rate=0.03, early_stopping_rounds=65, seed=42)
	model.fit(
	    X=X_train,
	    y=y_train,
	    group=qids_train,
	    eval_set=[(X_eval, y_eval)],
	    eval_group=[qids_eval], 
	    eval_at=[5],
		)
#elif test_or_cv=='CV':
	#	splitted=np.split(df_train_whole['srch_id'].unique())
	#srch_ids=dict()
	
#%% Final  Part
del df, df_eval,df_train,qids_eval, X_eval, X_train, y_eval, y_train # delete from memory after training to prevent runtime crashes and restarts 
#del df_test, X_test, results]
df_test_trimmed_nonna=df_test.drop(columns=['srch_id']+columns_to_drop+['Target']).fillna(0)

results = model.predict(df_test_trimmed_nonna)

df_results=pd.DataFrame(index=df_test_trimmed_nonna.index,data=results,columns=['Prop'])

df_res=pd.merge(df_test[['srch_id','prop_id']],df_results,left_index=True,right_index=True,how='left')
df_res['Prop'].fillna(0,inplace=True)

del df_test_trimmed_nonna, df_results,results

df_final=df_res.sort_values(['srch_id','Prop'],ascending=[True,False])
df_final['srch_id']=df_final['srch_id']-1000000
df_final.drop(columns='Prop').to_csv('sub71_PK_v3.csv',index=False)
	
#%% Extras
ax = lgbm.plot_importance(model)
fig = ax.figure
fig.set_size_inches(20,20)


#%%
lgbm.plot_split_value_histogram(model, 'price_usd_norm_prop_id')
lgbm.plot_metric(model)
