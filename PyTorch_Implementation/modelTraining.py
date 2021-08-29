# model training

# imports
import torch
import torch.nn as nn
import dataLoading
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import modelDefinition

validation = False   #specify whether validation shall be done  

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")     #check if gpu is available for training

print("Creating Model and Loading Trainingdata")
DenseNet = modelDefinition.DenseNet(in_channels=2, layer_size=[6,6,8,7], growth_rate=12, additional_neurons = 8).to(device=device)  #initialize DenseNet - Adapt Parameters!!!
DenseNet = DenseNet.float()     #convert parameters of network to float

#DenseNet.load_state_dict(torch.load(r"E:\Uni\Uni u. Schule\Uni\Semester 6\Software Projekt\AIT_Racing\AIT_Racing\PyTorch_Implementation\trained_desnenet.pth"))

if torch.cuda.is_available():
    DenseNet = DenseNet.cuda()  #load densenet to gpu if available

DenseNet.train()    #activate train mode for Densenet - Batchnorm activated & allows to pass batches


mean_r = 0    #initialize mean value for red input pixel normalization
mean_g = 0    #initialize mean value for green input pixel normalization
std_r = 0     #initialize standard deviation value for red input pixel normalization
std_g = 0     #initialize standard deviation value for green input pixel normalization

loader = DataLoader(dataLoading.transformed_dataset, batch_size=len(dataLoading.transformed_dataset), num_workers=0)    #create a dataset which contains all elements of dataLoading.dataset, in order to calculate mean and std

print("Calculating Mean & Std for Green & Red Pixles")     

for i_batch, sample_batched in enumerate(loader):   #calculate meand and std of the whole training set but individually for each channel and save values in variables
    mean_r = sample_batched['image'][:,0,:,:].mean()
    std_r = sample_batched['image'][:,0,:,:].std()
    mean_g = sample_batched['image'][:,1,:,:].mean()
    std_g = sample_batched['image'][:,1,:,:].std()

print("Mean Value red: ", mean_r,", Mean Value green: ", mean_g)
print("Standard Deviation red: ", std_r, ", Standard Deviation green: ", std_g)  #!make sure to use these values during validation / test!

transform = transforms.Compose(     #create transform operation which substracts mean for each image tensor and divides by std and also contains normal transpose trainformation for image
    [dataLoading.convert_data,
     transforms.Normalize(mean=[mean_r, mean_g], std=[std_r, std_g])])

normalized_data = dataLoading.SteeringCommands(csv_file=dataLoading.csv_path, root_dir=r"F:\Meine Programme\OpenAI_RaceGame\Training_Data\training_v2\training", transform=transform)    #create new normalized training set

normalized_loader = DataLoader(normalized_data, batch_size=32, shuffle=True, num_workers=0)    #load normalized training set - set hyperparameter batch_size

"""
for i_batch, sample_batched in enumerate(normalized_loader):
    print(i_batch, sample_batched['image'].size(), sample_batched['input_data'].size(),
          sample_batched['commands'].size())
    print(sample_batched['image'][0])
    print("Tpye Image: ",type(sample_batched['image'][0]))
    print("Type Commands:, ", type(sample_batched['commands'][0]))
"""

criterion = nn.BCELoss()    #defines loss function - Loss function used here is binary cross entropy loss (CEL for sigmoid)
if torch.cuda.is_available():
    criterion = nn.BCELoss().cuda()
optimizer = optim.Adam(DenseNet.parameters(), lr=0.001, betas=(0.9, 0.999)) #define optimizer - set hyper parameters lr and betas

nr_epochs = 30   #hyperparameter - determnies how many times the whole training set gets looped through the Neural Network

if(True):
    print("Starting Training")
    
    for epoch in range(nr_epochs):      #loop which trainis the NN with the normalized dataset "nr_epochs" times. - set hyperparameter nr_epochs 
        
        running_loss = 0    #variable to calculate runnning loss (only for output in console)
        
        for i, data in enumerate(normalized_loader):        #loop which goes through whole normalized dataset once
            images = data['image']                          #split up data dictionary in images, commands(steering, acceleration,brake) and inputs(speed, abs, gyroscope, steering)
            commands = data['commands']
            inputs = data['input_data']
            if torch.cuda.is_available():           #transfer tensors to gpu if available 
                commands = commands.cuda()
                inputs = inputs.cuda()
                images = images.cuda()
            
            optimizer.zero_grad()       #resets all gradients to zero 
            
            outputs = DenseNet(images.float(), inputs.float())  #calculate output of one batch, input tensors have to consist of float numbers
            loss = criterion(outputs.float(), commands.float())     #calculate loss, therefore convert function inputs (output of NN and labels) to float
            loss.backward()     #calculate gradient for each parameter based on loss -> dloss/dx
            
            optimizer.step()    #adapts values of NN
            
            
            running_loss += loss.item()     #add loss of this batch to running loss in order to calculate mean loss later on
            if i % 50 == 49:    # print every 50 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 50))   #print running loss on screen
                running_loss = 0.0  #reset running loss
                
        print("Epoch ", epoch + 1, " finished")
    
    print('Finished Training')


normalized_loader_validation = DataLoader(normalized_data, batch_size=1, shuffle=True, num_workers=0)


if(validation == True):
    print("Starting Validation")
    
    running_loss = 0
    
    DenseNet.eval()
    
    with torch.no_grad():
        for i, data in enumerate(normalized_loader_validation):        
            images = data['image']                          #split up data dictionary in images, commands(steering, acceleration,brake) and inputs(speed, abs, gyroscope, steering)
            commands = data['commands']
            inputs = data['input_data']
            if torch.cuda.is_available():           #transfer tensors to gpu if available 
                commands = commands.cuda()
                inputs = inputs.cuda()
                images = images.cuda()
            
            outputs = DenseNet(images.float(), inputs.float())
            commands = torch.squeeze(commands, 0)   #remove unnessecary dimension form commands tensor size:[1,4] -> size:[4]
            loss = criterion(outputs.float(), commands.float())   
            running_loss += loss.item()
                                       
            if i % 100 == 99:    # print every 50 mini-batches
                print('loss: %.3f' %
                      (running_loss / 100))
                print("Sample Tensor Output:", outputs)
                print("Sample Desired Output:", commands)
                running_loss = 0.0  
            
    print("Finished Validation")
#save model weights
PATH = './trained_desnenet.pth'         #path were is parameters are stored

print("Parameters Saved")
torch.save(DenseNet.state_dict(), PATH)     #save current state dict of model