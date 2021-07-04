# img_viewer.py

from predict_copy import predict

from random import uniform
import PySimpleGUI as sg
import os.path

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

def animate(i):
    xs=[]
    ys=[]
    for pick in dataForPlot:
        if len(dataForPlot) > 1:
            x,y = pick
            xs.append(x)
            ys.append(y)
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
        sg.Text("Preprocessed dataForPlot folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),
        sg.Button('Predict')
    ],
    [
        sg.T(""), 
        sg.Text("Choose a model: "), 
        sg.In(size=(25, 1), enable_events=True, key="-MODEL-"), 
        sg.FileBrowse(key="-MODEL-")
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
        '''
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
            dataLocation = folder
            
        except:
            file_list = []
        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".hdf5"))
        ]
        window["-MODEL-"].update(fnames)
        '''
    elif event == "-MODEL-":  # A file was chosen from the listbox
        fileLocation = values["-MODEL-"]
        ###try:
            ##fileLocation = os.path.join(
             ##   values["-MODEL-"]
            ##)
            #window["-TOUT-"].update(filename)
            #window["-IMAGE-"].update(filename=filename)

        ###except:
        ###    pass
    elif event == 'Predict':
        score_vid = []
        #------DANEDOWYKRESU--------
        for z in range(0,30):
            score_vid = uniform(0, 1)
            dataForPlot.append((z,score_vid))
        #score_vid = predict(fileLocation, dataLocation)
        print(dataLocation, fileLocation)


window.close()