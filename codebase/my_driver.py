# -*- coding: utf-8 -*-
"""
Easiest continuous control task to learn from pixels, a top-down racing
environment.
Discrete control is reasonable in this environment as well, on/off
discretization is fine.
State consists of STATE_W x STATE_H pixels.
The reward is -0.1 every frame and +1000/N for every track tile visited, where
N is the total number of tiles visited in the track. For example, if you have
finished in 732 frames, your reward is 1000 - 0.1*732 = 926.8 points.
The game is solved when the agent consistently gets 900+ points. The generated
track is random every episode.
The episode finishes when all the tiles are visited. The car also can go
outside of the PLAYFIELD -  that is far off the track, then it will get -100
and die.
Some indicators are shown at the bottom of the window along with the state RGB
buffer. From left to right: the true speed, four ABS sensors, the steering
wheel position and gyroscope.
To play yourself (it's rather fast for humans), type:
python gym/envs/box2d/car_racing.py
Remember it's a powerful rear-wheel drive car -  don't press the accelerator
and turn at the same time.
Created by Oleg Klimov. Licensed on the same terms as the rest of OpenAI Gym.
"""

import modelDefinition 
import torch   
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

counter = 0

def load_model():
    # initialize the model
    DenseNet = modelDefinition.DenseNet(in_channels=2, layer_size=[6, 6, 8, 7], growth_rate=12,
                                             additional_neurons=8)
    DenseNet.load_state_dict(torch.load("repo/model_weights/trained_densenetepoch110.pth")['model_state_dict'])
    DenseNet.eval()
    
    if torch.cuda.is_available():
        DenseNet = DenseNet.cuda()  #load densenet to gpu if available

    # initialize the container
    container = []

    return DenseNet, container


def driver(img, step, model, container):
    global counter
    mean_r = 1.1384
    mean_g = 194.3558
    std_r = 16.9998
    std_g = 108.5659
    transformed = transforms.Normalize(mean=[mean_r, mean_g], std=[std_r, std_g])

    commands = [0.0, 0.0, 0.0]
    inputs, image = preprocess(img)
    image = np.asarray(image)
    image = image[:, :, 0:2]

    speed = inputs["speed"]
    gyro = inputs["gyroscope"]
    inputs = [
        inputs["speed"],
        inputs["abs1"],
        inputs["abs2"],
        inputs["abs3"],
        inputs["abs4"],
        inputs["left"],
        inputs["right"],
        inputs["gyroscope"],
    ]

    inputs = np.asarray(inputs)
    inputs = inputs.astype('float').reshape(-1)
    inputs = torch.from_numpy(inputs)

    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image)
    image = transformed(image.float())
    image = torch.unsqueeze(image, 0)

    if torch.cuda.is_available():           #transfer tensors to gpu if available
            inputs = inputs.cuda()
            image = image.cuda()

    with torch.no_grad():
        output = model(image.float(), inputs.float())

    container.append(output)

    if (output[0] > 0.6):  # if left key activated  (0.7)
        commands[0] -= 0.7
    if (output[1] > 0.6):  # if right key activated (0.7)
        commands[0] += 0.7
    if (output[2] > 0.91):  # if up key activated (0.9)
        commands[1] = 0.9
        output[3] = 0.0
    if (output[3] > 0.5):  # if down key activated (0.5)
        commands[2] = 0.8

    if speed > 3:
        commands[1] = 0

    if counter < 30:
        commands[0] = 0
    if counter < 10:
        commands[1] = 1.0
    if (speed <= 1 and gyro == 0):
        commands[1] = 1
        commands[2] = 0

    counter += 1

    return commands, container  # list of [steering, acceleration, brake]


def preprocess(state):
    # initialization of values
    res = {}
    # verical
    res["speed"] = 0
    res["abs1"] = 0
    res["abs2"] = 0
    res["abs3"] = 0
    res["abs4"] = 0
    # horizontal
    res["left"] = 0
    res["right"] = 0
    res["gyroscope"] = 0

    for y in range(8):
        if np.array_equal((state[93 - y][13]), [255, 255, 255]):
            res["speed"] += 1
        if np.array_equal((state[93 - y][17]), [0, 0, 255]):
            res["abs1"] += 1
        if np.array_equal((state[93 - y][20]), [0, 0, 255]):
            res["abs2"] += 1
        if np.array_equal((state[93 - y][22]), [51, 0, 255]):
            res["abs3"] += 1
        if np.array_equal((state[93 - y][24]), [51, 0, 255]):
            res["abs4"] += 1

    for y in range(24):
        # starting x coordinate of horizontal bar
        green = 47
        red = 71
        if np.array_equal((state[89][green + y]), [0, 255, 0]):
            res["right"] += 1
        elif np.array_equal((state[89][green - y]), [0, 255, 0]):
            res["left"] += 1
                
        # find center
        if np.array_equal((state[89][red + y + 1]), [255, 0, 0]):
            res["gyroscope"] += 1
        if np.array_equal((state[89][red - y]), [255, 0, 0]):
            res["gyroscope"] -= 1
            
    return res, show_processed_image(state)


def show_processed_image(img_arr, image_name="image", show=False):
    # creating a image object
    image = Image.new('RGB', (96, 84), color='red')
    # result bool map
    res = []
    x = 0
    res.append([])
    # put pixel on image plane
    for row in img_arr[:84]:
        y = 0
        for pixel in row:
            if pixel[0] == pixel[2] and pixel[1] > 200 and pixel[0] < pixel[1]:
                col = (0, 255, 0)
                res[x].append(False)
            else:
                col = (0, 0, 0)
                res[x].append(True)
            image.putpixel((y, x), (col))
            y += 1
        x += 1
        res.append([])

    # paint the car
    for x in range(46, 50):
        for y in range(67, 77):
            if is_car_coordinate(x, y):
                image.putpixel((x, y), (255, 255, 0))

    return image


def is_car_coordinate(x, y):
    # car is from x = 46; y = 67  to x = 49; y = 76, except y = 70 - 71 && (x = 46 || x == 49)
    return (y != 70 and y != 71) or (x != 46 and x != 49)