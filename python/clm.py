from statsmodels.miscmodels.ordinal_model import OrderedModel
import pandas as pd
import numpy as np
from pandas.api.types import CategoricalDtype
from glob import glob
import time

files = glob("/home/iyori/work/gam/ordinal_regression/orca/datasets2/*/*/matlab/")
files += glob("/home/iyori/work/gam/ordinal_regression/orca/datasets2/*/*/*/matlab/")
print(files)
for f in files:
    metric_df = pd.DataFrame(columns=['datanum', "duration","train_acc", "train_mae", "test_acc", "test_mae"])
    acc = 0
    mae = 0
    train_files = glob(f + "train*")
    train_files.sort()
    datanum = 0
    for trainf in train_files:
        print(f.split("/")[-3], trainf.split(".")[1])
        testf = trainf.split("train")[0] + "test" + trainf.split("train")[1]
        traindf = pd.read_csv(trainf, delimiter=" ", header=None)
        testdf = pd.read_csv(testf, delimiter=" ", header=None)
        trainx = traindf.iloc[:,:-1]
        trainy = traindf.iloc[:,-1]
        testx = testdf.iloc[:,:-1]
        testy = np.array(testdf.iloc[:,-1])
        Q = trainy.max()
        y_type = CategoricalDtype(categories=list(range(1,Q+1)), ordered=True)
        trainycat = trainy.astype(y_type)
        trainy = np.array(trainy)

        print(Q, trainycat.dtype)
        try:
            mod_log = OrderedModel(trainycat,
                            trainx,
                            distr='logit')
        except Exception as e:
            print(trainf, e)
            continue
        now = time.time()
        res_log = mod_log.fit(method='bfgs')
        duration = time.time()-now
        num_of_thresholds = Q-1
        # print(res_log.summary())
        print(mod_log.transform_threshold_params(res_log.params[-num_of_thresholds:]))

        predicted = res_log.model.predict(res_log.params, exog=trainx)
        predictedy = predicted.argmax(axis=1)+1
        n = len(predictedy)
        localtrainacc = np.sum(trainy == predictedy)/n
        localtrainmae = np.sum(np.abs((trainy-predictedy)))/n

        predicted = res_log.model.predict(res_log.params, exog=testx)
        predictedy = predicted.argmax(axis=1)+1
        n = len(predictedy)
        localtestacc = np.sum(testy == predictedy)/n
        localtestmae = np.sum(np.abs((testy-predictedy)))/n
        acc += localtestacc
        mae += localtestmae
        datanum += 1
        metric_df = metric_df.append({"datanum":trainf.split(".")[1], "duration":duration,"train_acc": localtrainacc , "train_mae": localtrainmae, "test_acc": localtestacc, "test_mae": localtestmae }, ignore_index = True)
        print(trainf.split(".")[1], duration, localtrainacc, localtrainmae, localtestacc, localtestmae)
    if datanum==0:
        print(f.split("/")[-3], "no valid data")
        continue
    metric_df.sort_values("datanum", inplace=True)
    metric_df.to_csv(f'clmexp/{f.split("/")[-3]}.csv')
    acc/=datanum
    mae/=datanum
    print("++++++++++++++++++++++++++++++++++++++++++++")
    print(acc, mae)
        
# mod_prob = OrderedModel(data_diam['cut'],
#                         data_diam[['volume', 'price', 'carat']],
#                         distr='probit')