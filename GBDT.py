import pandas as pd
import lightgbm as lgb
import gc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#输出准确率
def do_model_metric(y_true, y_pred, y_pred_prob):
    from sklearn.metrics import roc_auc_score,accuracy_score
    print("AUC: {0:.3}".format(roc_auc_score(y_true=y_true, y_score=y_pred_prob[:,1])))
    print("Accuracy: {0}".format(accuracy_score(y_true=y_true, y_pred=y_pred)))


#current_service的values转化为0-11
def changeValues(current_service):
    current_service_series = [89950166, 89950167, 89950168, 90063345, 90109916, 90155946, 99999825, 99999826, 99999827,
                              99999828, 99999830]
    for k in range(len(current_service)):
        for i in range(11):
            if (current_service[k] == current_service_series[i]):
                current_service[k] = i
    return current_service


#读取数据
train_data=pd.read_csv("train_0928.csv")
test_data=pd.read_csv("test_one_hot.csv")
train_sz = train_data.shape[0]
current_service =train_data.current_service.values
current_service=changeValues(current_service)
train_data.drop(columns=['current_service','contract_type_8','user_id'],inplace=True)
test_data.drop(columns=['user_id'],inplace=True)
combineDf = pd.concat([train_data, test_data], axis=0, ignore_index=True)
del test_data
gc.collect()
lgb_feats=train_data.columns.values.tolist()
#print(test_data.info())
X_train,X_validation,y_train,y_validation=train_test_split(train_data,current_service,test_size=0.4)
train_data=lgb.Dataset(train_data,label=current_service)
validation_data=lgb.Dataset(X_validation,label=y_validation)
del X_train,X_validation,y_train,y_validation
gc.collect()


params={
    'learning_rate':0.2,
    'lambda_l1':0.1,
    'lambda_l2':0.2,
    'max_depth':6,
    'num_leaves':10,
    'objective':'multiclass',
    'num_class':11,
}
clf=lgb.train(params,train_data,valid_sets=[validation_data])
del validation_data,train_data
gc.collect()


# LR + GBDT
#得到叶节点编号 Feature Transformation
gbdt_feats_vals = clf.predict(combineDf, pred_leaf=True)
gbdt_columns = ["gbdt_leaf_indices_" + str(i) for i in range(0, gbdt_feats_vals.shape[1])]
combineDf = pd.concat([combineDf, pd.DataFrame(data=gbdt_feats_vals, index=range(0, gbdt_feats_vals.shape[0]),columns=gbdt_columns)], axis=1)

# onehotencoder(gbdt_feats)
origin_columns = combineDf.columns
for col in gbdt_columns:
    combineDf = pd.concat([combineDf, pd.get_dummies(combineDf[col], prefix=col)],axis=1)
gbdt_onehot_feats = [col for col in combineDf.columns if col not in origin_columns]
# 恢复train, test
train = combineDf[:train_sz]
test = combineDf[train_sz:]
del combineDf
gc.collect()

lr_gbdt_feats = lgb_feats+ gbdt_onehot_feats
lr_gbdt_model = LogisticRegression(penalty='l2', C=1)
lr_gbdt_model.fit(train[lr_gbdt_feats],current_service)

print("Train................")
do_model_metric(y_true=current_service, y_pred=lr_gbdt_model.predict(train[lr_gbdt_feats]), y_pred_prob=lr_gbdt_model.predict_proba(train[lr_gbdt_feats]))

print("Test..................")
#do_model_metric(y_true=test['current_service'], y_pred=lr_gbdt_model.predict(test[lr_gbdt_feats]), y_pred_prob=lr_gbdt_model.predict_proba(test[lr_gbdt_feats]))

