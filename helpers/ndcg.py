from sklearn.metrics import ndcg_score
import pandas as pd
import numpy as np



def ndcg_eval(df_1,df_2, show_difference=False):
	"""
	

	Parameters
	----------
	df_1 : Dataframe including srch_id column with ordered prop_id, like in the submission file.
	df_2 : Dataframe for the same srch_id, but including also click_bool and booking_bool.

	Returns
	-------
	Average ndcg score

	"""
	ndcg=0
	for srch_id in df_1['srch_id'].unique():
		df_1_temp=df_1.loc[df_1['srch_id']==srch_id,'prop_id']
		df_2_temp=df_2.loc[df_2['srch_id']==srch_id,['prop_id','booking_bool','click_bool']]
		df_temp=pd.merge(df_1_temp,df_2_temp,how='left',on='prop_id')
		df_temp['Rel']=0
		df_temp.loc[df_temp['click_bool']==1,'Rel']=1
		df_temp.loc[df_temp['booking_bool']==1,'Rel']=5
		y_score=df_temp['Rel'].values
		n_5=sum(y_score==5)
		n_1=sum(y_score==1)
		y_true=np.append(np.append(np.repeat(5,n_5),np.repeat(1,n_1)),np.repeat(0,len(y_score)-n_1-n_5))
		if show_difference:
    		print('true and predicted')
			print(y_true)
			print(y_score)
		ndcg+=ndcg_score([y_true],[y_score])
	return(ndcg/len(df_1['srch_id'].unique()))
