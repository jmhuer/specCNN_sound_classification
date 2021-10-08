# specCNN_sound_classification
Pytorch workspace for sound classification projects. Few model options available, including a standard CNN Classifier, Pre-trained AlexNet, Pre-trained ResNet50, and other standard architectures.

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

base = "Data/UrbanSound8K/audio/"
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
# Step 3: Live inference model

Test your model. 

```
python -m Live_inference.live_inference 
```

if no model given as argument, newest model in Model/saved_models is used
(optional) Prediction default to running majority of a window_size.
