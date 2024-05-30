




**Data**
Data folder contains metadata, datasets imported in Notebooks fromt the repositry, sample images and information realted to datsets.



# 1. Malaria detection using Convolutional Neural Networks

This robust malaria detection model is built using Convolutional Neural Networks (CNN). Leveraging powerful Python libraries such as TensorFlow, PIL, and scikit-learn, I've developed a model capable of accurately identifying parasites (infections) in blood smear images. This comprehensive approach includes preprocessing with NumPy and pandas, data visualization with Matplotlib and Seaborn, and efficient model training utilizing advanced CNN layers like Conv2D, MaxPooling2D, and Dense layers. 

Explore the intricacies of my approach, from data augmentation to early stopping, as I aim for precise diagnosis through cutting-edge image classification.
https://github.com/hari255/DeepLearning/blob/main/_Malaria_Detection.ipynb

## Convlutional Neural Networks.

A CNN is a type of neural network that's really good at looking at pictures and figuring out what's in them. This specific model is set up to look at images that are 64x64 pixels with 3 color channels (red, green, blue) and decide between categories.

![image](https://github.com/hari255/Neural-Networks/assets/59302293/6be05b5d-6bf4-4a45-b2dd-fb604d538060)



## Data Exploration

**Data set size:** 25000 samples of images which includes parasites and non-parasite cells. Below image is plotted using Python on the dataset.

<img width="460" alt="image" src="https://github.com/hari255/Neural-Networks/assets/59302293/247ea202-56a1-45a4-a959-bec0e40e9bee">

+ Infected cells have some form of disturbances within the cell with a dark color formation.

+ UnInfected cells have uniform color throughout the image.

To accurately identify parasites in cell images and train our model to distinguish them, we must focus on the key aspects that differentiate these categories. This involves analyzing and understanding the unique characteristics and features that set each parasite apart. By doing so, we can ensure that our model is well-equipped to recognize and classify the various types of parasites accurately.


+ **Average Uninfected Image**
<img width="389" alt="image" src="https://github.com/hari255/Neural-Networks/assets/59302293/992a7520-26a7-419e-84b3-ce3601fc4f64">

+ **Average Infected Image**
<img width="392" alt="image" src="https://github.com/hari255/Neural-Networks/assets/59302293/e950ebaf-1cfe-4b4a-8e4f-14bf80222e60">


**The mean image for both parasitized and uninfected are pretty much same because, the diffence between these two images are very small (infection is the only difference). The average image is obviously the larger part of it and how it seems most likely is the idea.**

## Data Transformation

In the data transformation stage, I've tried to use multiple techniques used in Image procesing and Computer Vision and expereimenting with my dataset. These techinques are useful in different stages of Model building.

| Technique | Description |
| ------ | ----------- |
| RGB to HSV  | To seperate image brightness from color information |
| Gaussian smoothing | To remove noise from the image |
| Data Augmentation   | To slightly alter the image and generate new one |


**Converting RGB to HSV using OpenCV**

The purpose of converting RGB images to HSV (Hue, Saturation, Value) using OpenCV is to facilitate more effective image processing and analysis. The HSV color space separates image intensity (brightness) from color information, which can be particularly useful for various image processing tasks. 

In the HSV color space, it's wasy to seperate colors based on their hue. This is useful for segmenting objects in an image based on color. This property helps us identify and differentiate te infected cell images.

`Python code to convert the images to HSV using Open CV`

``` py
  import cv2

  gfx=[]   # to hold the HSV image array
  
  for i in np.arange(0, 100, 1):
  
   a = cv2.cvtColor(train_images[i], cv2.COLOR_BGR2HSV)
   
   gfx.append(a)
   
  gfx = np.array(gfx)

  viewimage = np.random.randint(1, 100, 5)
  
  fig, ax = plt.subplots(1, 5, figsize = (18, 18))
  
  for t, i in zip(range(5), viewimage):
  
   Title = train_labels[i]
   
   ax[t].set_title(Title) 
   
   ax[t].imshow(gfx[i])
   
   ax[t].set_axis_off()
   
   fig.tight_layout()

``` 


**Utilizing Gaussian Blurring**

Gaussina Blurring or Smoothing is a technique that helps in removing noise from an image. It uses Gaussian kernel to weigh the neighboring pixels based on a Gaussian distribution.

<img width="252" alt="image" src="https://github.com/hari255/Neural-Networks/assets/59302293/1165c5af-51d5-4173-950e-cc82a864bbb5">

+ (x,y) are the coordinates of the pixel.
+ σ is the standard deviation of the Gaussian distribution, which controls the amount of blurring.

*After Gaussian smoothing, the images looks like below.


``` py

model3 = Sequential()

# Build the model here
model3.add(Conv2D(filters = 32, kernel_size = 2, padding = "same", activation = "relu", input_shape = (64, 64, 3)))

model3.add(MaxPooling2D(pool_size = 2))

model3.add(Dropout(0.2))

model3.add(Conv2D(filters = 32, kernel_size = 2, padding = "same", activation = "relu"))

model3.add(MaxPooling2D(pool_size = 2))

model3.add(Dropout(0.2))

model3.add(Conv2D(filters = 32, kernel_size = 2, padding = "same", activation = "relu"))

model3.add(MaxPooling2D(pool_size = 2))

model3.add(Dropout(0.2))

model3.add(Flatten())

model3.add(Dense(512, activation = "relu"))

model3.add(Dropout(0.4))

model3.add(Dense(2, activation = "softmax")) # 2 represents output layer neurons

# Use this as the optimizer
adam = optimizers.Adam(learning_rate = 0.001)

model3.compile(loss = 'binary_crossentropy' , optimizer = adam, metrics = ['accuracy'])

model3.summary()

```





---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




To benchmark my innovative approach, I've conducted a comprehensive performance analysis against the renowned VGG16 image-detection model. The findings not only highlight the superiority of my model but also shed light on key metrics that set it apart.
VGG16 model; https://keras.io/api/applications/vgg/ 

Team, Keras. (2016). Keras Documentation: VGG16 and VGG19. https://keras.io/api/applications/vgg/

# 2. Recommedation systems based on Amazon user's purchase history


Hey there! 🚀 Welcome to the Amazon product Recommender project! Imagine having a buddy who knows exactly what you'd like to buy online - that's what we're creating here!

**The Story:**
You know how sometimes there's just too much stuff online, and it's hard to decide what to pick? Well, our Amazon Magic Recommender is like a superhero that helps you find cool stuff you'll love! It's like having a friend who knows your favorite things.

**How it Works:**
Big companies like Amazon use special tricks to help you find the best things to buy. They use super-smart computer programs (we call them algorithms) that look at what you liked before and suggest similar awesome stuff. It's like magic for shopping!

**Our Mission:**
We're on a mission to build our own magic recommender! We're using a special set of data from Amazon, where people gave ratings to different gadgets. We're turning these ratings into a secret code so our magic recommender can learn what people like and recommend cool things to them.

So, get ready for a magical journey where we make online shopping super fun and easy! 🌟✨ Get ready to discover some awesome stuff made just for you! 🚀🛍️
 
# 3. Leveraging Auto ML using H2O

**What is Auto ML?**

Automated machine learning (AutoML) is the process of automating the tasks of applying machine learning to real-world problems.

AutoML, short for Automated Machine Learning, refers to the process of automating various aspects of machine learning model development and deployment.

AutoML typically automates the following aspects of the machine learning workflow:

1. Data Preprocessing
2. Feature Selection
3. hyperparameter Tuning
4. Evaluation
5. Deployment
... Uh, that's a lot.! Also, it can interpret results. :)

**In this project, we just go through the basic steps like setting up H20 in a notebook and running through a ML problem by leveraging H2O AutoML doing the work for us.**

H2O offers many solutions to the day-to-day problems that data scientists and machine learning engineers face. It saves a lot of time during pre-processing, tuning, and evaluation. 
https://h2o.ai/company/


Thank you! :)
