# img_viewer.py


from re import X
import PySimpleGUI as sg
import os.path
from predict import predict_fun
import os

import numpy as np


from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PySimpleGUI as sg
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

#Plot function
dataForPlot = []
fig, ax = plt.subplots()
matplotlib.use("TkAgg")
score_vid = []

def animate(i):
    score_vid = np.load('score_vid.npy')
    xs=[]
    ys=[]
    i = 0
    for pick in score_vid:
        if len(score_vid) > 1:
            x = i
            y = score_vid[i]
            xs.append(x)
            ys.append(y)
            i+=1
    ax.cla()
    ax.set_title("Anomaly detection plot")
    ax.set_xlabel("Time")
    ax.set_ylabel("Prediction")
    ax.grid()
    ax.plot(xs,ys)
    plt.plot(xs, ys)      
    fig.canvas.draw() 

ani = animation.FuncAnimation(fig, animate, interval=1000)



sg.theme('DarkAmber')   # Add a touch of color
# First the window layout in 2 columns
file_list_column = [
    [
        sg.Text("Choose a video file with testing data"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FileBrowse(),
        sg.Button('Predict')
    ],
    [
        sg.T(""), 
        sg.Text("Choose a folder with model: "), 
        sg.In(size=(25, 1), enable_events=True, key="-MODEL-"), 
        sg.FolderBrowse(key="-MODEL-")
    ]
]
# For now will only show the name of the file that was chosen

image_viewer_column = [
    [sg.Text("Anomaly detection plot")],
    [sg.Canvas(key="-CANVAS-")],
]

# ----- Full layout -----
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column),
        
    ]
]
window = sg.Window("Abnormal event detection", layout, finalize=True)

# Link matplotlib to PySimpleGUI Graph
canvas = FigureCanvasTkAgg(fig, window["-CANVAS-"].Widget)
plot_widget = canvas.get_tk_widget()
plot_widget.grid(row=0, column=0)

# Run the Event Loop
fileLocation = 'deklaracja'
dataLocation = 'deklaracja'
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    # Folder name was filled in, make a list of files in the folder
    if event == "-FOLDER-":
        dataLocation = values["-FOLDER-"]
     
    elif event == "-MODEL-":  # A file was chosen from the listbox
        fileLocation = values["-MODEL-"]

    elif event == 'Predict':
        
        #------Function for prediction and generating score for the plot--------
        predict_fun(fileLocation, dataLocation)
        

window.close()