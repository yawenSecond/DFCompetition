#coding = utf-8
import  pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

#current_service的values转化为0-11
def changeValues(current_service):
    current_service_series = [89950166, 89950167, 89950168, 90063345, 90109916, 90155946, 99999825, 99999826, 99999827,
                              99999828, 99999830]
    for k in range(len(current_service)):
        for i in range(11):
            if (current_service[k] == current_service_series[i]):
                current_service[k] = i
    return current_service

data=pd.read_csv('train_new.csv')
print(data.info())
current_service =data.current_service.values
current_service=changeValues(current_service)
#pay_num=train_data.pay_num.values
data.drop(columns=['current_service','user_id','service_type','is_mix_service','online_time','many_over_bill','contract_type','contract_time','is_promise_low_consume','net_service','complaint_level','gender'],inplace=True)
X_train,X_validation,y_train,y_validation=train_test_split(data,current_service,test_size=0.4)
train_data=lgb.Dataset(data,label=current_service)
validation_data=lgb.Dataset(X_validation,label=y_validation)

params={
    'learning_rate':0.05,
    'lambda_l1':0.1,
    'lambda_l2':0.2,
    'max_depth': 6,
    'objective':'multiclass',
    'num_class':11,
}
clf=lgb.train(params,train_data,valid_sets=[validation_data])
pred = clf.predict(data, pred_leaf=True)
#print(pred)
for i in range(pred.shape[1]):
    if(i==0):
        result = pd.DataFrame(pd.get_dummies(pred[i], prefix=i))
    else:
        result= pd.concat([result, pd.get_dummies(pred[i],prefix=i)],axis=1)

result.to_csv('result.csv')