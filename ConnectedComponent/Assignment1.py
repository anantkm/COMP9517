# -*- coding: utf-8 -*-
"""
Created on Sat Oct 3 9:00:00 2020

@author:Anant Krishna Mahale
zID: 5277610
"""

#......IMPORT .........
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np 
import cv2 
import argparse
import os


def switchWhiteBlack(input_image):
    for y, row in enumerate(input_image): 
        for x, pixel_value in enumerate(row):
            if pixel_value == 255:
                input_image[y,x] = 0
            else:
                input_image[y,x] = 255
    return input_image

def task_1_interMediateResults(input_image,image_name,output_folder,iteration_counter,t):
    input_image[input_image<t] = 0
    input_image[input_image>=t] = 255
    input_image = switchWhiteBlack(input_image)
    fig = plt.figure()
    plt.axis('off')
    fig = plt.imshow(input_image,cmap='gray', vmin=0, vmax=255)
    plt.title("Threshold Value =" + str(t),fontweight=8,pad='2.0',fontsize = 5)
    plt.savefig(output_folder+image_name+'_Task1_'+str(iteration_counter)+'.png',bbox_inches='tight',dpi=1000)


def apply_median_filter(input_image):
    filtered_values = []
    image_width = len(input_image[0])
    image_height = len(input_image)
    filtered_image = np.zeros((image_height, image_width), dtype=np.int16)
    for x in range(image_height):
        for y in range(image_width):
            for z in range(6):
                if x + z - 3 < 0 or x + z - 3 > image_height - 1:
                    for i in range(6):
                        filtered_values.append(0)
                else:
                    if y + z - 3 < 0 or y + 3 > image_width - 1:
                        filtered_values.append(0)
                    else:
                        for j in range(6):
                            filtered_values.append(input_image[x + z - 3][y + j - 3])

            filtered_values.sort()
            median_value = len(filtered_values) // 2
            filtered_image[x][y] = filtered_values[median_value]
            filtered_values = []
    return filtered_image   


def modifyLabelDict(label_dict):
    
    for key,value in label_dict.items():
        label_dict[key] = min(value)
    
    for key,value in label_dict.items():
        label_dict[key] = label_dict[value]
    
    return label_dict    

def find_neighbour_value(image, x, y):
    neighbour_labels = set()

    # North(Upper) neighbour
    if y > 0: 
        north_lebel = image[y-1,x]
        if north_lebel > 0: #neighbour has a value
            neighbour_labels.add(north_lebel)

    # West(Left) neighbour
    if x > 0: 
        west_label = image[y,x-1]
        if west_label > 0: #neighbour has a value
            neighbour_labels.add(west_label)


    # North-West(Right Diagonal) neighbour
    if x > 0 and y > 0: 
        northwest_label = image[y-1,x-1]
        if northwest_label > 0: #neighbour has a value
            neighbour_labels.add(northwest_label)

    # North-East (Left Diagonal) neighbour
    if y > 0 and x < len(image[y]) - 1: 
        northeast_label = image[y-1,x+1]
        if northeast_label > 0: #neighbour has a value
            neighbour_labels.add(northeast_label)

    return neighbour_labels

def connected_component_labelling(input_image):
    #find the height and width of the image.
    image_width = len(input_image[0])
    image_height = len(input_image)

    # initialising numpy array with integers.
    labelled_image = np.zeros((image_height, image_width), dtype=np.int16)
    current_label = 1  #intial label value
    label_dict = {}

    #Pass_1: labelling the image and it's equivalances 
    for y, row in enumerate(input_image):
        for x, pixel_value in enumerate(row):
            if pixel_value == 0: #background
                pass 
            else:
                labels = find_neighbour_value(labelled_image, x, y) #find neighbour values and equivalances

                if not labels: # if it's empty, use new label.  
                    labelled_image[y,x] = current_label
                    label_dict[current_label] = [current_label]
                    current_label = current_label + 1 #increment the current label.    

                else: #add it to the equivalance dictionary. 
                    smallest_label = min(labels)
                    labelled_image[y,x] = smallest_label
                    if len(labels) > 1:
                        for label in labels:
                            for value in labels:
                                label_dict[label].append(value)
    
    modfied_label_dict = modifyLabelDict(label_dict)

    
    #Pass_2: replacing the labels with smallest Equivalent 
    for y, row in enumerate(labelled_image):        
        for x, pixel_value in enumerate(row):
            if pixel_value > 0: # Foreground pixel
                labelled_image[y,x] = modfied_label_dict[pixel_value]
    
    print('Number of Rice Kernals = '+ str(len(set(modfied_label_dict.values()))))
    rice_kernels = len(set(modfied_label_dict.values()))
    return labelled_image,rice_kernels


def task1(input_image,image_name,output_folder):
    t = 60      #random threshold value. 
    epsilon = 0.01
    threshold_values = []
    iteration_counter_list = []
    iteration_counter = 0 
    t_new = 0
    diff = abs(t - t_new)
    while diff >= epsilon:
        m_zero = input_image[input_image<t].mean()
        m_one = input_image[input_image>=t].mean()
        t_new = (m_zero+m_one)/2
        diff = abs(t - t_new)
        t = t_new
        iteration_counter += 1
        iteration_counter_list.append(iteration_counter)        
        threshold_values.append(t)
        # task_1_interMediateResults(input_image.copy(),image_name,output_folder,iteration_counter,t) #uncomment this if intermediate results are needed.

    fig = plt.figure()
    print('Threshold Value = '+str(t))
    plt.scatter(iteration_counter_list,threshold_values,color = 'blue')
    # plt.plot(iteration_counter_list,threshold_values,color = 'blue')
    plt.xticks(iteration_counter_list)
    fig.suptitle('Threshold Value = '+str(t))
    plt.xlabel('Number of Iterations')
    plt.ylabel('Threshold Value')
    plt.show()  
    input_image[input_image<t] = 0
    input_image[input_image>=t] = 255
    output_image = switchWhiteBlack(input_image.copy())
    plt.clf()
    fig = plt.imshow(output_image,cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.title("Threshold Value =" + str(t),fontweight=8,pad='2.0',fontsize = 5)
    plt.savefig(output_folder+image_name+'_Task1.png',bbox_inches='tight',dpi=1000)
    return input_image 

def task2(input_image,image_name,output_folder):
    filtered_image = apply_median_filter(input_image)
    output_image = switchWhiteBlack(filtered_image.copy())
    labelled_image, rice_count = connected_component_labelling(filtered_image)
    plt.clf()
    fig = plt.imshow(output_image,cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.title("No. of Rice Kernels = " + str(rice_count),fontweight=8,pad='2.0',fontsize = 5)
    plt.savefig(output_folder+image_name+'_Task2.png',bbox_inches='tight',dpi=1000)
    return labelled_image
    
def task3(labelled_image,image_name,output_folder,min_area):
    stats = {}
    for y, row in enumerate(labelled_image): 
        for x, pixel_value in enumerate(row):
            if pixel_value > 0: # Foreground pixel
                if pixel_value in stats:
                    stats[pixel_value] += 1
                else:
                    stats[pixel_value] = 1    
    
    damaged_rice_kernals = 0
    for key,value in stats.items():
        if value < min_area:
            stats[key] = 0
            damaged_rice_kernals+=1
        else:
            stats[key] = 255    

    for y, row in enumerate(labelled_image):        
        for x, pixel_value in enumerate(row):
            if pixel_value > 0: # Foreground pixel
                labelled_image[y,x] = stats[pixel_value]
    
    output_image = switchWhiteBlack(labelled_image.copy())
    print("No. of Damaged Kernels = " + str(damaged_rice_kernals))
    percent_damage = damaged_rice_kernals/(len(stats))*100
    print("Percentage of damaged kernals = " + str(percent_damage) + '%')
    fig = plt.imshow(output_image,cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.title("Percentage of damaged kernals = " + str(percent_damage) + '%',fontweight=8,pad='2.0',fontsize = 5)
    plt.savefig(output_folder+image_name+'_Task3.png',bbox_inches='tight',dpi=1000)
    

    
my_parser = argparse.ArgumentParser()
my_parser.add_argument('-o','--OP_folder', type=str,help='Output folder name', default = 'OUTPUT')
my_parser.add_argument('-m','--min_area', type=int,action='store', required = True, help='Minimum pixel area to be occupied, to be considered a whole rice kernel')
my_parser.add_argument('-f','--input_filename', type=str,action='store', required = True, help='Filename of image ')
# Execute parse_args()
args = my_parser.parse_args()
output_folder = args.OP_folder
image_name = os.path.splitext(os.path.basename(args.input_filename))[0]
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

input_image_1 = cv2.cvtColor(cv2.imread(args.input_filename), cv2.COLOR_BGR2GRAY)
task_1_output = task1(input_image_1,image_name,str(output_folder)+'/')
task_2_output = task2(task_1_output,image_name,str(output_folder)+'/')
task3(task_2_output,image_name,str(output_folder)+'/',args.min_area)
