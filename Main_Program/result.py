import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from cvzone.ClassificationModule import Classifier
_DIR_ = "test_data"
classifier = Classifier("keras_model.h5","labels.txt")
true_counter = 0
false_counter = 0
arr1 = []
arr_1 = []
arr2 = []
arr_2 = []
time_data = []
counter = 0
sum = 0
labels = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","W","X","Y","Z"]
for folder in os.listdir(_DIR_):
    for file in os.listdir(os.path.join(_DIR_,folder)):
        file = cv2.imread(os.path.join(_DIR_,folder,file))
        stime = time.time()
        prediction,index = classifier.getPrediction(file,draw=None)
        etime = time.time()
        time_data.append(round((etime-stime)*1000))
        counter+=1
        sum+=round((etime-stime)*1000)
        try:
            if(labels[index] == folder):
                print("TRUE")
                true_counter +=1
                arr1.append(prediction[index])
                arr_1.append(true_counter)
            else:
                print("FALSE")
                false_counter +=1
                arr2.append(prediction[index])
                arr_2.append(false_counter)
        except:
            pass


figure, axis = plt.subplots(figsize = (8, 3))
axis.set_title(f"Average time = {round(sum/counter)} ms")
axis.hist(time_data, bins = [20, 40, 60, 80, 100],edgecolor = "black")  

data = {"TRUE" : true_counter,"FALSE" : false_counter}
a = list(data.keys())
b = list(data.values())
fig = plt.figure(figsize=(5,5))


plt.bar(a, b, color ='blue') 
plt.xlabel("TRUE POSITIVE TRUE NEGATIVE")
plt.ylabel("No. of test cases")
plt.title(f"{(true_counter/(false_counter+true_counter)) * 100}% is accuracy")
plt.show()



plt.plot(arr_1,arr1)
plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.title('True Positive Graph')
plt.show()


plt.plot(arr_2,arr2)
plt.xlabel('x - axis')
plt.title('False Positive Graph')
plt.show()