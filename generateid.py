import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# 
# train=pd.read_csv('./output.csv')
# test=pd.read_csv('./input/test.csv')


train=pd.read_csv('./blending/submission.csv')
test=pd.read_csv('./blending/samples/pred_blend.csv')



sub=test['id'].to_frame()
sub['target']=train['target'].to_frame()
print(sub.head())
sub.to_csv('submission_libffm_pure.csv', index=False, float_format='%.5f') 
# gc.collect()
sub.head(2)