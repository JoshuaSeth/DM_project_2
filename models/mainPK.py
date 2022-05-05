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


random_n=100000 #0 takes all of them
competitor_strategy='Keep'
orig_destination_distance_strategy='Drop'
visitors_strategy='Drop'
hotel_data_strategy='Keep'
normalization=(['prop_review_score','prop_location_score1','price_usd'],['srch_id','prop_id','srch_booking_window','srch_destination_id','prop_country_id'])

plots=False



#%% Basic plotting for EDA
if plots:
	df = load_data()
	corr=df.drop(columns=['srch_id','visitor_location_country_id','date_time','site_id','prop_country_id','prop_id','srch_destination_id','gross_bookings_usd']).corr()
	
	# Set up the matplotlib figure
	f, ax = plt.subplots(figsize=(11, 9))
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
#df_all=pd.read_pickle('df_all.pkl')
comp_columns=['comp1_rate', 'comp1_inv',
      'comp1_rate_percent_diff', 'comp2_rate', 'comp2_inv',
      'comp2_rate_percent_diff', 'comp3_rate', 'comp3_inv',
      'comp3_rate_percent_diff', 'comp4_rate', 'comp4_inv',
      'comp4_rate_percent_diff', 'comp5_rate', 'comp5_inv',
      'comp5_rate_percent_diff', 'comp6_rate', 'comp6_inv',
      'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv',
      'comp7_rate_percent_diff', 'comp8_rate', 'comp8_inv',
      'comp8_rate_percent_diff']

if competitor_strategy=='Drop':
	df_trimmed1=df_all.drop(columns=comp_columns)
else:
	df_trimmed1=df_all.copy()
	df_trimmed1.loc[:,comp_columns]=df_trimmed1.loc[:,comp_columns].fillna(0)

del df_all
	
if orig_destination_distance_strategy=='Drop':
	df_trimmed1.drop(columns='orig_destination_distance',inplace=True)
	
if visitors_strategy=='Drop':
	df_trimmed1.drop(columns=['visitor_hist_starrating','visitor_hist_adr_usd'],inplace=True)
#msno.matrix(df_trimmed1)
df_trimmed1.isnull().sum().sort_values(ascending=False)

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
#df_trimmed2.to_pickle('df_trimmed2.pkl')
#%% Part III Train set
not_in_train=['gross_bookings_usd','click_bool','booking_bool','position']
target1='booking_bool'
target2='click_bool'

df_train1=df_trimmed2.drop(columns=[n for n in not_in_train if n != target1]).loc[df_trimmed2['Train']==1,:].dropna()
df_train2=df_trimmed2.drop(columns=[n for n in not_in_train if n != target2]).loc[df_trimmed2['Train']==1,:].dropna()


if random_n>0:
	srch_ids=sample(list(df_trimmed2['srch_id'].unique()),random_n)
	df_train1=df_train1.loc[df_train1['srch_id'].isin(srch_ids),:].drop(columns=['srch_id','date_time','Train'])
	df_train2=df_train2.loc[df_train2['srch_id'].isin(srch_ids),:].drop(columns=['srch_id','date_time','Train'])
else:
	df_train1=df_train1.drop(columns=['srch_id','date_time','Train'])
	df_train2=df_train2.drop(columns=['srch_id','date_time','Train'])



#%% Modelling part
param_grid = { 'n_estimators': [10, 25,50], 'max_depth': [2, 5,10] }
clf1=RandomForestClassifier(random_state=0,verbose=2)
grid_clf1 = GridSearchCV(clf1, param_grid, cv=5)
grid_clf1.fit(df_train1.drop(columns=target1),df_train1[target1])
rf1=grid_clf1. best_estimator_


clf2=RandomForestClassifier(random_state=0,verbose=2)
grid_clf2 = GridSearchCV(clf2, param_grid, cv=5)
grid_clf2.fit(df_train2.drop(columns=target2),df_train2[target2])
rf2=grid_clf2. best_estimator_



#Old
#rf1=RandomForestClassifier(max_depth=5,random_state=0,verbose=2)
#rf1.fit(df_train1.drop(columns=target1),df_train1[target1])

#rf2=RandomForestClassifier(max_depth=5,random_state=0,verbose=2)
#rf2.fit(df_train2.drop(columns=target2),df_train2[target2])

#%% Testing Part
df_test=df_trimmed2.drop(columns=not_in_train+['date_time','Train']).loc[df_trimmed2['Train']==0,:]

df_test.isnull().sum().sort_values(ascending=False)
df_test_trimmed_nonna=df_test.drop(columns='srch_id').fillna(0)
pred1=rf1.predict_proba(df_test_trimmed_nonna)
df_pred1=pd.DataFrame(index=df_test_trimmed_nonna.index,data=pred1,columns=['InvProp1','Prop1']).drop(columns='InvProp1')

pred2=rf2.predict_proba(df_test_trimmed_nonna)
df_pred2=pd.DataFrame(index=df_test_trimmed_nonna.index,data=pred2,columns=['InvProp2','Prop2']).drop(columns='InvProp2')

df_pred=pd.concat([df_pred1,df_pred2],axis=1)
df_pred['Prop']=df_pred['Prop1']*5+df_pred['Prop2']
df_pred.drop(columns=['Prop1','Prop2'],inplace=True)
df_res=pd.merge(df_test,df_pred,left_index=True,right_index=True,how='left')
df_res['Prop'].fillna(0,inplace=True)

del df_test_trimmed_nonna, df_pred, pred1,pred2, df_pred1, df_pred2

df_final=df_res.sort_values(['srch_id','Prop'],ascending=[True,False]).loc[:,['srch_id','prop_id']]
df_final['srch_id']=df_final['srch_id']-1000000
df_final.to_csv('sub71_PK_v3.csv',index=False)





#%% Old code testing
#df_test=load_data(caching=False,test=True)
#df_test_trimmed=df_test.loc[:,[n for n in df_train.columns if n != target1]]
#df_test_trimmed.isnull().sum().sort_values(ascending=False)
#df_test_trimmed_nonna=df_test_trimmed.dropna()
#pred=rf1.predict_proba(df_test_trimmed_nonna)
#df_pred=pd.DataFrame(index=df_test_trimmed_nonna.index,data=pred,columns=['Prop','InvProp']).drop(columns='InvProp')
#df_res=pd.merge(df_test,df_pred,left_index=True,right_index=True,how='left')
#df_res['Prop'].fillna(0,inplace=True)
#
#del pred,df_test_trimmed, df_test_trimmed_nonna, df_pred
#
#df_final=df_res.sort_values(['srch_id','Prop'],ascending=[True,False]).loc[:,['srch_id','prop_id']]
#df_final.to_csv('sub71_PK_v1.csv',index=False)
