import os
import cv2 as cv
import numpy as np
import json

train_folders = ['023.bulldozer','072.fire-truck','182.self-propelled-lawn-mower','192.snowmobile','224.touring-bike',
                 '251.airplanes-101','252.car-side-101']
train_folders_test=['bulldozer_tst','fire-truck_tst','self-propelled-lawn-mower_tst','snowmobile_tst','touring-bike_tst','airplanes_tst',
                    'car-side_tst']
sift = cv.xfeatures2d_SIFT.create()

def extract_local_features(path):
    img = cv.imread(path)

    kp = sift.detect(img)
    desc = sift.compute(img, kp)
    desc = desc[1]
    return desc

def encode_bovw_descriptor(desc, vocabulary):
    bow_desc = np.zeros((1, vocabulary.shape[0]))
    for d in range(desc.shape[0]):
        distances = np.sum((desc[d, :] - vocabulary) ** 2, axis=1)
        mini = np.argmin(distances)
        bow_desc[0, mini] += 1
    return bow_desc



# Extract Database
#train_descs = np.zeros((0, 128))
#for folder in train_folders:
#    files = os.listdir(folder)
#    for file in files:
#        path = os.path.join(folder, file)
#        desc = extract_local_features(path)
#       train_descs = np.concatenate((train_descs, desc), axis=0)

# Create vocabulary
#term_crit = (cv.TERM_CRITERIA_EPS, 30, 0.1)
#loss, assignments, vocabulary = cv.kmeans(train_descs.astype(np.float32), 50, None, term_crit, 1, 0)
#np.save('vocabulary2.npy', vocabulary)
 # Load vocabulary
vocabulary = np.load('vocabulary2.npy')

# Create Index
#img_paths = []
#train_descs = np.zeros((0, 128))
#bow_descs = np.zeros((0, vocabulary.shape[0]))
#for folder in train_folders:
#   files = os.listdir(folder)
#    for file in files:
#        path = os.path.join(folder, file)
#        desc = extract_local_features(path)
#        bow_desc = encode_bovw_descriptor(desc, vocabulary)
#
#        img_paths.append(path)
#        bow_descs = np.concatenate((bow_descs, bow_desc), axis=0)
#np.save('index2.npy', bow_descs)
#with open('index_paths2.txt', mode='w+') as file:
#   json.dump(img_paths, file)

# # Load Index
bow_descs = np.load('index2.npy')
with open('index_paths2.txt', mode='r') as file:
   img_paths = json.load(file)



# Search


svm = cv.ml.SVM_create()
svm.setType(cv.ml.SVM_C_SVC)
svm.setKernel(cv.ml.SVM_RBF)
svm.setTermCriteria((cv.TERM_CRITERIA_COUNT, 100, 1.e-06))

labels = np.array(['bulldozer' in a for a in img_paths], np.int32)
svm.trainAuto(bow_descs.astype(np.float32), cv.ml.ROW_SAMPLE, labels)

responses = np.zeros((0, 1))
for folder in train_folders_test:
    files = os.listdir(folder)

    for file in files:
        test_img = os.path.join(folder, file)

        desc = extract_local_features(test_img)
        bow_desc = encode_bovw_descriptor(desc, vocabulary)

        response = svm.predict(bow_desc.astype(np.float32), flags=cv.ml.STAT_MODEL_RAW_OUTPUT)
        responses = np.concatenate((responses, response[1]), axis=0)
response1=responses

labels = np.array(['fire-truck' in a for a in img_paths], np.int32)
svm.trainAuto(bow_descs.astype(np.float32), cv.ml.ROW_SAMPLE, labels)

responses = np.zeros((0, 1))
for folder in train_folders_test:
    files = os.listdir(folder)

    for file in files:
        test_img = os.path.join(folder, file)

        desc = extract_local_features(test_img)
        bow_desc = encode_bovw_descriptor(desc, vocabulary)

        response = svm.predict(bow_desc.astype(np.float32), flags=cv.ml.STAT_MODEL_RAW_OUTPUT)
        responses = np.concatenate((responses, response[1]), axis=0)

response2=responses

labels = np.array(['self-propelled-lawn-mower' in a for a in img_paths], np.int32)
svm.trainAuto(bow_descs.astype(np.float32), cv.ml.ROW_SAMPLE, labels)


responses = np.zeros((0, 1))
for folder in train_folders_test:
    files = os.listdir(folder)

    for file in files:
        test_img = os.path.join(folder, file)

        desc = extract_local_features(test_img)
        bow_desc = encode_bovw_descriptor(desc, vocabulary)

        response = svm.predict(bow_desc.astype(np.float32), flags=cv.ml.STAT_MODEL_RAW_OUTPUT)
        responses = np.concatenate((responses, response[1]), axis=0)


response3=responses

labels = np.array(['snowmobile' in a for a in img_paths], np.int32)
svm.trainAuto(bow_descs.astype(np.float32), cv.ml.ROW_SAMPLE, labels)

responses = np.zeros((0, 1))
for folder in train_folders_test:
    files = os.listdir(folder)

    for file in files:
        test_img = os.path.join(folder, file)

        desc = extract_local_features(test_img)
        bow_desc = encode_bovw_descriptor(desc, vocabulary)

        response = svm.predict(bow_desc.astype(np.float32), flags=cv.ml.STAT_MODEL_RAW_OUTPUT)
        responses = np.concatenate((responses, response[1]), axis=0)

response4=responses

labels = np.array(['touring-bike' in a for a in img_paths], np.int32)
svm.trainAuto(bow_descs.astype(np.float32), cv.ml.ROW_SAMPLE, labels)

responses = np.zeros((0, 1))
for folder in train_folders_test:
    files = os.listdir(folder)

    for file in files:
        test_img = os.path.join(folder, file)

        desc = extract_local_features(test_img)
        bow_desc = encode_bovw_descriptor(desc, vocabulary)

        response = svm.predict(bow_desc.astype(np.float32), flags=cv.ml.STAT_MODEL_RAW_OUTPUT)
        responses = np.concatenate((responses, response[1]), axis=0)


response5=responses

labels = np.array(['airplanes' in a for a in img_paths], np.int32)
svm.trainAuto(bow_descs.astype(np.float32), cv.ml.ROW_SAMPLE, labels)

responses = np.zeros((0, 1))
for folder in train_folders_test:
    files = os.listdir(folder)

    for file in files:
        test_img = os.path.join(folder, file)

        desc = extract_local_features(test_img)
        bow_desc = encode_bovw_descriptor(desc, vocabulary)

        response = svm.predict(bow_desc.astype(np.float32), flags=cv.ml.STAT_MODEL_RAW_OUTPUT)
        responses = np.concatenate((responses, response[1]), axis=0)


response6=responses

labels = np.array(['car-side' in a for a in img_paths], np.int32)
svm.trainAuto(bow_descs.astype(np.float32), cv.ml.ROW_SAMPLE, labels)


responses = np.zeros((0,1))
for folder in train_folders_test:
    files = os.listdir(folder)
    for file in files:
        test_img = os.path.join(folder, file)



        desc = extract_local_features(test_img)
        bow_desc = encode_bovw_descriptor(desc, vocabulary)

        response = svm.predict(bow_desc.astype(np.float32), flags=cv.ml.STAT_MODEL_RAW_OUTPUT)
        responses = np.concatenate((responses , response[1]), axis=0)

response7=responses

all_responses=np.concatenate((response1,response2,response3,response4,response5,response6,response7),axis=1)
print('result',all_responses)

sum=0
for i in range (0,4):
    min=2

    for j in range (0,6):
        if all_responses[i][j]<min :

            min=all_responses[i][j]
            k=j
    if k==0:
        sum=sum+1


for i in range (5,9):
    min=2

    for j in range (0,6):
        if all_responses[i][j]<min :

            min=all_responses[i][j]
            k=j
    if k==1:
        sum=sum+1


for i in range (10,14):
    min=2
    for j in range (0,6):
        if all_responses[i][j]<min :

            min=all_responses[i][j]
            k=j
    if k==2:
        sum=sum+1

for i in range (15,19):
    min=2
    for j in range (0,6):
        if all_responses[i][j]<min :

            min=all_responses[i][j]
            k=j
    if k==3:
        sum=sum+1


for i in range(20, 24):
    min = 2
    for j in range(0,6):
        if all_responses[i][j] < min:
            min = all_responses[i][j]
            k = j
    if k == 4:
        sum = sum + 1

for i in range(25, 29):
    min = 2
    for j in range(0,6):
        if all_responses[i][j] < min:
            min = all_responses[i][j]
            k = j
    if k == 5:
        sum = sum + 1

for i in range(30, 34):
    min = 2
    for j in range(0,6):
        if all_responses[i][j] < min:
            min = all_responses[i][j]
            k = j
    if k == 6:
        sum = sum + 1

SVM_result=(sum/35)*100
print('Το ποσοστό επιτυχίας χρησιμοποιώντας την μέθοδο SVM είναι', SVM_result, '%')

###############################################################################################################3
# Knn

sum_all = 0
for folder in train_folders_test:
    files = os.listdir(folder)
    sum_bull = 0
    sum_fire = 0
    sum_bike = 0
    sum_air = 0
    sum_car = 0
    sum_snow = 0
    sum_self = 0


    for file in files:
        test_img = os.path.join(folder, file)

        desc = extract_local_features(test_img)
        bow_desc = encode_bovw_descriptor(desc, vocabulary)
        K = 6
        distances = np.sum((bow_desc - bow_descs) ** 2, axis=1)
        sorted_ids = np.argsort(distances)

        for i in range(K):
            if 'bulldozer' in img_paths[sorted_ids[i]]:
                sum_bull = sum_bull+1
            elif 'fire-truck' in img_paths[sorted_ids[i]]:
                sum_fire +=1

            elif 'self-propelled' in img_paths[sorted_ids[i]]:
                sum_self +=1

            elif 'snowmobile' in img_paths[sorted_ids[i]]:
                sum_snow +=1

            elif 'bike' in img_paths[sorted_ids[i]]:
                sum_bike +=1

            elif 'airplanes' in img_paths[sorted_ids[i]]:
                sum_air +=1
            else:
                sum_car+=1
        if sum_bull>sum_fire and sum_bull>sum_self and sum_bull>sum_snow and sum_bull>sum_bike and sum_bull>sum_air and sum_bull>sum_car:
            if 'bulldozer' in folder:
                sum_all=sum_all+1
        elif sum_fire>sum_self and sum_fire>sum_snow and sum_fire>sum_bike and sum_fire>sum_air and sum_fire>sum_car:
            if 'fire-truck' in folder:
                sum_all=sum_all+1
        elif sum_self>sum_snow and sum_self>sum_bike and sum_self>sum_air and sum_self>sum_car:
            if 'self-propelled-lawn-mower' in test_img:
                sum_all=sum_all+1
        elif sum_snow>sum_bike and sum_snow>sum_air and sum_snow>sum_car:
            if 'snowmobile' in folder:
                sum_all=sum_all+1
        elif sum_bike>sum_air and sum_bike>sum_car:
            if 'touring-bike' in folder:
                sum_all=sum_all+1
        elif sum_air>sum_car:
            if 'airplanes' in folder:
                sum_all=sum_all +1
        elif sum_car>sum_air:
            if 'car-side' in folder:
                sum_all=sum_all+1
        else:
            sum_all=sum_all+1
knn_result=(sum_all/35)*100
print('Το αποτέλεσμα με την μέθοδο Knn ειναι ',knn_result ,' %')

