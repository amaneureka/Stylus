## Stylus
An offline handwriting recognition hardware and tensorflow based model implementation that will let you type with ease. Right now it is supporting english alphabets and numerics. That are 62 symbols!

#### Model
##### Structure
- Convolution Layer
 - 16 3x3 Filters
- Max Pool
 - 3x3 Same Padding
- Rectified Linear Unit
- Convolution Layer
 - 10 3x3 Filters
- Max Pool
 - 2x2 Same Padding
- Rectified Linear Unit
- Inception 16 filters
- Fully Connected
 - 1 Hidden Layer, 500 Neurons
 - TanH activation
- Softmax Layer
##### Can I edit the model?
Yes, Model structure is stored seperately in ```models/cnn.py```. ```Train.py``` expects following function from a model script

```def create_network(img_height, img_width, num_classes)```

And it should return ```x, y, y_true, optimizer```
##### How can train the new model?
```train.py``` has been designed to aid this by using preconfigured values and training method. You can expect a detailed explanation of different parameters by typing ```train.py --help```.
examples
- ```train.py --show```
  - Show loaded training data
- ```train.py --train```
  - Train model
- ```train.py --restore save/validations```
  - Test model
#### Dataset
##### About
We're using public dataset [Chars74K](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/)
- It contains 62 classes [A-Za-z0-9]
- It has 55 samples per class
- All images are of fixed resolution 1200x900
##### How to download dataset?
You can download and unzip dataset by running ```dataset.sh``` shell script.
##### Preprocessing
We're are preprocessing dataset because of various reasons
- We can reduce dimensions
- We can eliminate empty spaces
- We don't need 3 channels
- Samples are not properly scaled
- Too big to fit into memory and train on home pc
##### Stages of normalization
- Bounding box calculation
- Samples croping
- Reducing channels
- Scale to fit properly in maximum bounding box
- Dimensions Reduction
##### How to preprocess data ?
You can preprocess data after downloading dataset by executing ```python normalize.py```
- It expects that system has all the dependencies installed
  - ```pip3 install -r requirements.txt```
  - ```bash ./dataset.sh```

It will create two files ```normalized-train.bin``` & ```normalized-val.bin``` under ```dataset``` directory.
##### Normalized dataset
![Normalized dataset](http://i.imgur.com/FwmpAHn.png)

#### Project
This is a part of Acadmic Project Report under **Practical Training** (COE-320) at [**NSIT**](www.nsit.ac.in).

#### Contributors
- Arko Gupta
- Aryan Singh
- [Aman Priyadarshi](https://twitter.com/amaneureka)

#### License
MIT &copy; Aman Priyadarshi

