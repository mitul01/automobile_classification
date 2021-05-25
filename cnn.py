from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
# Initialising the CNN
classifier = Sequential()
# Step 1 - Convolution1
classifier.add(Conv2D(32,(3,3), input_shape = (64, 64, 3), activation = 'relu'))
# Step 2 - Pooling1
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Step 3 - Convolution2
classifier.add(Conv2D(32, (3,3), activation = 'relu'))
# Step 4 - Pooling2
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Step 5 - Flattening
classifier.add(Flatten())
# Step 4 - Full connection Network
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 3, activation = 'softmax'))
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('G:\Data',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('G:\Data',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')

model = classifier.fit_generator(training_set,
                         steps_per_epoch = 1000,
                         epochs = 1,
                         validation_data = test_set,)


classifier.save("model_new.h5")
print("Saved model to disk")

print(training_set.class_indices)

import sys
import copy
import time

s = time.time()
stack = []  # stack-of-current-nodes
visited = []  # list-of-visited-nodes
start = [[2, 8, 3], [1, 6, 4], [7, 0, 5]]  # start-node , 0 indicates null position
goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]  # goal-node , 0 indicates null position
n = len(start)  # size-of-puzzle


def finish(start, goal):
    if start == goal:
        return 1  # Return true when start node becomes equal to goal node
    return 0


def find_null(start):  # A function to find null space in puzzle
    for i in range(n):
        for j in range(n):
            if start[i][j] == 0:  # 0 indicates null position
                return ([i, j])  # return when empty(null) space found in puzzle


# For all the functions below:- # 1) pos stores the location of null space in puzzle
# 2) 0 indicates null position
def up(start, pos):  # This Function is used to move null space upwards
    if pos[0] > 0:  # For upward movement row should be atleast greater than 0
        temp = copy.deepcopy(start)
        temp[pos[0]][pos[1]] = temp[pos[0] - 1][pos[1]]
        temp[pos[0] - 1][pos[1]] = 0
        return (temp)
    else:
        return (start)


def down(start, pos):  # This Function is used to move null space downwards
    if pos[0] < n - 1:  # For downward movement row should be less than n-1
        temp = copy.deepcopy(start)
        temp[pos[0]][pos[1]] = temp[pos[0] + 1][pos[1]]
        temp[pos[0] + 1][pos[1]] = 0
        return (temp)
    else:
        return (start)


def right(start, pos):  # This Function is used to move null space rightwards
    if pos[1] < n - 1:  # For rightward movement column should be less than n-1
        temp = copy.deepcopy(start)
        temp[pos[0]][pos[1]] = temp[pos[0]][pos[1] + 1]
        temp[pos[0]][pos[1] + 1] = 0
        return (temp)
    else:
        return start


def left(start, pos):  # This Function is used to move null space leftwards
    if pos[1] > 0:  # For leftward movement column should be greater than 0
        temp = copy.deepcopy(start)
        temp[pos[0]][pos[1]] = temp[pos[0]][pos[1] - 1]
        temp[pos[0]][pos[1] - 1] = 0
        return (temp)
    else:
        return (start)


import copy
import sys
import time
import random as r

s = time.time()  # start time of program
# define necessary things like current queue , visited queue , start & goal node , size of puzzel
queue = []
visited = []
start = [[2, 8, 3], [1, 6, 4], [7, 0, 5]]
goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
n = len(start)  # size-of-puzzle


def finish(start, goal):
    if start == goal:
        return 1  # Return true when start node becomes equal to goal node
    return 0


def find_null(start):  # A function to find null space in puzzle
    for i in range(n):
        for j in range(n):
            if start[i][j] == 0:  # 0 indicates null position
                return ([i, j])  # return when empty(null) space found in puzzle


def heuristic_func(current):  # Heuristic function used is == No.of Misplaced tiles
    misplaced = 0
    for i in range(len(current)):
        for j in range(len(current)):
            if current[i][j] != goal[i][j]:
                misplaced += 1
    return misplaced  # returns no.of misplaced tiles


def simple_hill_climbing(start, goal_state):
    i, j = find_null(start)  # store the position of null in i,j
    queue.append(start)
    count = 0
    parent_heuristic = heuristic_func(start)  # first heuristic values

    while len(queue) != 0:
        print("Parent Heuristic :", parent_heuristic)
        count += 1
        if finish(queue[0], goal_state):
            print("Goal State Reached")
            break  # break the loop when goal state reached
        else:
            flag = 0
            temp = copy.deepcopy(queue[0])
            visited.append(temp)
            start = queue.pop(0)  # Remove the first value in queue as per FIFO rule
            i, j = find_null(start)  # find the position of null vlaue
            if i > 0:  # Condition for upward movement
                (start[i][j], start[i - 1][j]) = (
                start[i - 1][j], start[i][j])  # swap null value with value above it->Upward movement
                if start not in visited:
                    child_heuristic = heuristic_func(start)
                    print("Child Heuristic :", child_heuristic)
                    if child_heuristic < parent_heuristic:  # Hill climbing progress when child heuristic is lower than parent
                        parent_heuristic = child_heuristic
                        queue.append(copy.deepcopy(start))
                        flag = 1
                if flag == 1:
                    continue
                else:
                    start[i][j], start[i - 1][j] = (start[i - 1][j], start[i][j])

            if j > 0:  # Condition for leftward movement
                (start[i][j], start[i][j - 1]) = (
                start[i][j - 1], start[i][j])  # swap null value with value to left of it->Left movement
                if start not in visited:
                    child_heuristic = heuristic_func(start)
                    print("Child Heuristic :", child_heuristic)
                    if child_heuristic < parent_heuristic:  # Hill climbing progress when child heuristic is lower than parent
                        parent_heuristic = child_heuristic
                        queue.append(copy.deepcopy(start))
                        flag = 1
                if flag == 1:
                    continue
                else:
                    (start[i][j], start[i][j - 1]) = (start[i][j - 1], start[i][j])

            if i < n - 1:  # Condition for downward movement
                (start[i][j], start[i + 1][j]) = (
                start[i + 1][j], start[i][j])  # swap null value with value below it->Downward movement
                if start not in visited:
                    child_heuristic = heuristic_func(start)
                    print("Child Heuristic :", child_heuristic)
                    if child_heuristic < parent_heuristic:  # Hill climbing progress when child heuristic is lower than parent
                        parent_heuristic = child_heuristic
                        queue.append(copy.deepcopy(start))
                        flag = 1
                if flag == 1:
                    continue
                else:
                    (start[i][j], start[i + 1][j]) = (start[i + 1][j], start[i][j])

            if j < n - 1:  # Condition for rightward movement
                (start[i][j], start[i][j + 1]) = (
                start[i][j + 1], start[i][j])  # swap null value with value to right of it->rightward movement
                if start not in visited:
                    child_heuristic = heuristic_func(start)
                    print("Child Heuristic :", child_heuristic)
                    if child_heuristic < parent_heuristic:  # Hill climbing progress when child heuristic is lower than parent
                        parent_heuristic = child_heuristic
                        queue.append(copy.deepcopy(start))
                        flag = 1
                if flag == 1:
                    continue
                else:
                    (start[i][j], start[i][j + 1]) = (start[i][j + 1], start[i][j])

        elif len(queue) == 0:  # Empty queue indicated no possible child nodes left.
        print("\nBetter child Heuristic value not found.\nProcess Exited.\nSolution Reached")
        break


print("Number of moves :", count)

simple_hill_climbing(start, goal)
print("Execution Time :", round((time.time() - s) * 1000, 2), "ms")
