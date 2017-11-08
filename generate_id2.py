import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import numpy as np
# test files

# test_xgb = pd.read_csv(base_path + 'test_sub_xgb.csv')
# test_lgb = pd.read_csv(base_path + 'test_sub_lgb.csv')
# test_dnn = pd.read_csv(base_path + 'test_dnn_predictions.csv')
# test_up = pd.read_csv(base_path + 'test_submission.csv')
# test_cat = pd.read_csv(base_path + 'test_catboost_submission.csv')
# test_kin = pd.read_csv(base_path + 'test_uberKinetics.csv')
# test_gp = pd.read_csv(base_path + 'test_gpari.csv')

# test=pd.read_csv(base_path + 'test.csv')


# test = pd.concat([test, 
#                    test_xgb[['target']].rename(columns = {'target' : 'xgb'}),
#                    test_lgb[['target']].rename(columns = {'target' : 'lgb'}),
#                    test_dnn[['target']].rename(columns = {'target' : 'dnn'}),
#                    test_up[['target']].rename(columns = {'target' : 'up'}),
#                    test_cat[['target']].rename(columns = {'target' : 'cat'}),
#                    test_kin[['target']].rename(columns = {'target' : 'kin'}),
#                    test_gp[['target']].rename(columns = {'target' : 'gp'})                   
#                   ], axis = 1)


# train_cols = ['xgb', 'lgb', 'dnn', 'up', 'cat', 'kin', 'gp']







# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# # import xgboost as xgb
# import glob
# import csv
# from scipy import stats
base_path = './input/stacking/'
# df1=pd.read_csv(base_path + 'rank_avg_12.csv')

# df1=pd.read_csv(base_path + 'rank_avg_12_experiment.csv')

df1=pd.read_csv(base_path + 'pred_ffm.csv')
df2 = pd.read_csv(base_path + 'rank_avg_LATEST_experiment_19.csv')

# # df3=pd.read_csv('./blending/samples/pred_gpari.csv')

# # df5=pd.read_csv('./blending/samples/pred_blend.csv')
# # df4=pd.read_csv('./blending/samples/pred_285.csv')

# # df7=pd.read_csv('./blending/samples/pred_chance.csv')
# # df4=pd.read_csv('./blending/samples/pred_ffm.csv')
# # df9=pd.read_csv('./blending/samples/pred_uber.csv')




# train = pd.read_csv('./input/train.csv')
















df2.columns = [x+'_2' if x not in ['id'] else x for x in df2.columns]
# df4.columns = [x+'target_4' if x not in ['id'] else x for x in df4.columns]

# df5.columns = [x+'target_5' if x not in ['id'] else x for x in df5.columns]
# df6.columns = [x+'target_6' if x not in ['id'] else x for x in df6.columns]
# df7.columns = [x+'target_7' if x not in ['id'] else x for x in df7.columns]
# df8.columns = [x+'target_8' if x not in ['id'] else x for x in df8.columns]
# df9.columns = [x+'target_9' if x not in ['id'] else x for x in df9.columns]

# # blend = pd.merge(df1, df3, df2, how='left', on='id')

blend = df1.merge(df2,how='left',on='id')
# # .merge(df5,how='left',on='id').merge(df6,how='left',on='id').merge(df7,how='left',on='id').merge(df8,how='left',on='id').merge(df9,how='left',on='id')



# print(blend)
for c in df1.columns:
    if c != 'id':
        blend[c] = 0.3*(blend[c])+(0.7*blend[c+'_2'])
#         blend[c] = stats.hmean([blend[c],blend[c+'target_2'],blend[c+'target_3'],blend[c+'target_4']])
#         # blend[c] = 9*(blend[c] * (blend[c+'target_2']) * (blend[c+'target_3']) * (blend[c+'target_4'])
#         #     *(blend[c+'target_5']) * (blend[c+'target_6'])*(blend[c+'target_7'])
#         #     * (blend[c+'target_8'])*(blend[c+'target_9']))/(blend[c] + (blend[c+'target_2']) + (blend[c+'target_3']) + (blend[c+'target_4'])
#         #     +(blend[c+'target_5']) + (blend[c+'target_6'])+(blend[c+'target_7'])
#         #     + (blend[c+'target_8'])+(blend[c+'target_9']))
blend = blend[df1.columns]
# blend['target'] = (np.exp(blend['target'].values) - 1.0).clip(0,1)
blend.to_csv('blend_test1_ffm_096best2.csv', index=False)













# # with open('result.csv', 'w', newline='') as f_output:
# #     csv_output = csv.writer(f_output)
# #     dfs = []

# #     for filename in glob.glob('./blending/samples/pred_*.csv'):
# #         print (filename)
# #         # for filename in filenames:
# #         dfs.append(pd.read_csv(filename)['target'])

# #     print(dfs)

#         # with open(filename, newline='') as f_input:
#         #     csv_input = csv.reader(f_input)
#         #     header = next(csv_input)
#         #     averages = []
#         #     print(*csv_input)
#             # for col in zip(*csv_input):
#             #   print(col)
#                 # averages.append(sum(int(x) for x in col) / len(col))

#         # csv_output.writerow([filename] + averages)


# # train=pd.read_csv('./output.csv')
# # test=pd.read_csv('./input/test.csv')


# # train=pd.read_csv('./blending/submission_common.csv')
# # test=pd.read_csv('./sub10.csv')



# # sub=test['id'].to_frame()
# # sub['target']=train['Target'].to_frame()
# # print(sub.head())
# # sub.to_csv('submission_common_final.csv', index=False, float_format='%.5f') 
# # # gc.collect()
# # sub.head(2)