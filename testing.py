# %%
# %%
import pandas as pd
from tensorflow.keras.preprocessing import image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, multilabel_confusion_matrix, roc_curve, confusion_matrix, accuracy_score, classification_report
from tensorflow.keras.models import load_model
import cv2 as cv


# %%

SIZE=256
test = pd.read_csv('./dataset/RFMiD_2_Testing_labels.csv')

df_test = pd.DataFrame(test)
test_labels=df_test['id'].values


# %%

test_image = []
for i  in tqdm(test_labels):
    img = image.load_img('./dataset/Test_set/'+str(i)+'.jpg',target_size=(SIZE,SIZE,3))
    img = image.img_to_array(img)
    img = img/255
    test_image.append(img)
X_test = np.array(test_image)


# %%
from tensorflow.keras.models import load_model

model=load_model(f'./models/RFMiD2/DenseNet2 retrained/Densenet2_BCE_bestmodel_2.hdf5')

y_test = np.array(test.drop(['id'],axis=1))
_, acc = model.evaluate(X_test, y_test)
# print("Accuracy = ", (acc * 100.0), "%" )

y_pred = model.predict(X_test)
roc=roc_auc_score(y_test, y_pred, average='micro')
# print("AUC= ",roc)

y_test_df=y_test.argmax(axis=1)
y_pred_df=y_pred.argmax(axis=1)
# print(y_test_df)
# print(y_pred_df)

tn, fp, fn, tp = confusion_matrix(y_test_df, y_pred_df).ravel()
PR=tp/(tp+fp)
RC=tp/(tp+fn)
F1=2*(PR*RC)/(PR+RC)
ACC=(tp+tn)/(tp+fp+tn+fn)



print(f"\n:::::::::::Best::::::::::::::::::\nPrecision: ",PR,"\nRecall: ",RC,"\nF1-Score",F1,"\nAccuracy:",ACC,"\nROC:",roc)
print("True Positive: ",tp,"False Positive: ",fp,"True Negative: ",tn,"False Negative: ",fn)
print(f'\n*********************************************************************************************************************')




