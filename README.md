# specCNN_sound_classification
Pytorch workspace for sound classification projects. Few model options available, including a standard CNN Classifier, Pre-trained AlexNet, Pre-trained ResNet15, and other standard architectures.

Code is ready for cpu or gpu training.

Automatic launch of tensorboard (deleting old logs everytime we run train)

First decide classes (Model/models.py)
```
classes = ["class1", "class2", "class3" ...., "class20"]
```

# (Optional) Step 1a: Collect Data
Record microphone stream continously using 3 sec chucks 

```
python -m Data.rec_data
```

# (Optional) Step 1b: Breakdown existing dataset
If you have a folder with each different sound you would like to classify such as
```
Data/UrbanSound8K/audio/
 ───folder1
│   │   example1.wav
│   │   example1.wav      
└───folder2
│   │   example1.wav
│   │   example1.wav      
└───folder3
│   │   example1.wav
│   │   example1.wav      
└───folder4 
...
...

```
THis assumes the classes are 

```
classes = ["folder1", "folder2", "folder3" ...., "folder4"]
```

Specify path: 
base = "Data/UrbanSound8K/audio/"

Specify target folder path: 
target = "Data/Audio/"

```
python -m Data.manipulate_dataset
```

# Step 2: Train Model

Select model and loss

For example:
```
model = MyResNet()
```
```
loss = torch.nn.CrossEntropyLoss().to(device)
```
Finally run: 
```
python -m Model.train 
```
See arg options for logging and other hyperparameter specifications 

# Step 3: Live inference model

Test your model. 

```
python -m Live_inference.live_inference 
```

if no model given as argument, newest model in Model/saved_models is used
(optional) Prediction default to running majority of a window_size.
