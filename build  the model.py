import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.keras import models,layers
import matplotlib.pyplot as plt
import cv2
image_size=150
batch_size=32
Epochs=25

# load the train images dataset
train_dataset=tf.keras.preprocessing.image_dataset_from_directory(
    'C:/Users/Lenovo/Downloads/Data Set/intel/seg_train/seg_train',
    shuffle=True,
    image_size=(image_size,image_size),
    batch_size=batch_size)
class_names=train_dataset.class_names
print("class_names: "+str(class_names))

# load the test images dataset
test_dataset=tf.keras.preprocessing.image_dataset_from_directory(
    'C:/Users/Lenovo/Downloads/Data Set/intel/seg_test/seg_test',
    shuffle=True,
    image_size=(image_size,image_size),
    batch_size=batch_size)
class_names=test_dataset.class_names
print("class_names: "+str(class_names))

#reshuffle for improve its performance 
train_dataset=train_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset=test_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

# feature scaling 
rescale=tf.keras.Sequential([
    layers.Resizing(image_size,image_size),
    layers.Rescaling(1.0/255)])


# layer for data augmentation
data_augmentation=tf.keras.Sequential([
    layers.RandomFlip('horizontal_and_vertical'),
    layers.RandomRotation(0.2)])

input_shape=(batch_size,image_size,image_size,3)
num_of_classes=len(class_names)

# model
model=models.Sequential([
    rescale,
    data_augmentation,
    layers.Conv2D(16,(3,3),activation='relu',input_shape=input_shape),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(32,(3,3),activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128,(3,3),activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    
    layers.Dense(128,activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_of_classes,activation='softmax')
])
model.build(input_shape=input_shape)
print(model.summary())

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# train the model
validate=model.fit(train_dataset,epochs=Epochs,batch_size=batch_size,verbose=1,validation_data=test_dataset)

# print the epochs value of the accuracy
print(validate.validate['accuracy'])

# plot the accuracy and loss
acc=validate.validate=['accuracy']
val_acc=validate.validate=['val_acc']
loss=validate.validate=['loss']
val_loss=validate.validate=['val_loss']


plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(range(Epochs),acc,label='Train accuracy')
plt.plot(range(Epochs),val_acc,label='Validation accuracy')
plt.legend(loc='lower right')
plt.title('Train and validation accuracy')

plt.subplot(1,2,2)
plt.plot(range(Epochs),loss,label='Train accuracy')
plt.plot(range(Epochs),val_loss,label='Validation accuracy')
plt.legend(loc='upper right')
plt.title('Train and validation accuracy')

plt.show()


model.save(filepath='C:/Users/Lenovo/Downloads/Data Set/intel/model.h5')
channels=3
model=tf.keras.model.load_model('C:/Users/Lenovo/Downloads/Data Set/intel/model.h5')
print(model.summary())

def predict_image(model,img):
    img_array=tf.keras.preprocessing.image.img_to_array(img)
    img_array=tf.expand_dims(img_array,0)
    
    predictions=model.predict(img_array)
    
    result=predictions[0]
    result_index=np.argmax(result)
    
    predicted_class=class_names[result_index]
    confidence=round(100*np.max(result),2)
    
    return predicted_class,confidence

# predict on a single image
img_path='C:/Users/Lenovo/Downloads/Data Set/intel/seg_pred/seg_pred/70.jpg'
original_image=cv2.imread(img_path)
test_image=image.load_img(img_path,target_size=(image_size,image_size))
print(type(test_image))

test_image=image.img_to_array(test_image)
print(type(test_image))
print(test_image.shape)

# run the predict function
predicted_class,confidence=predict_image(model,test_image)

print(predicted_class)
print(confidence)

#show the result with the image
#resize the image(larger)
scale_percent=300
width=int(originalImage.shape[1]*scale_percent/100)
height=int(originalImage.shape[0]*scale_percent/100)
dim=(width,height)
resized=cv2.resize(originalImage,dim,interpolation=cv2.INTER_AREA)
resized=putText(resized,predicted_class,(10,100),cv2.FONT_HERSHEY_COMPLEX,1.6,(255,0,0),3,cv2.LINE_AA)

cv2.imshow('img',resized)
cv2.waitkey(0)













