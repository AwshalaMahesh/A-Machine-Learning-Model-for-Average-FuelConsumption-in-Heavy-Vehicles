import tkinter as tk
from tkinter import messagebox, filedialog, Text, Button, Label, Scrollbar  # Import Scrollbar
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

main = tk.Tk()
main.title("Average Fuel Consumption")
main.geometry("1300x1200")

filename = ""
train_x, test_x, train_y, test_y = None, None, None, None
balance_data = None
model = None
ann_acc = None
testdata = None
predictdata = None

def importdata(): 
    global balance_data, filename
    balance_data = pd.read_csv(filename)
    balance_data = balance_data.abs()
    return balance_data 

def splitdataset(balance_data):
    global train_x, test_x, train_y, test_y
    X = balance_data.values[:, 0:7] 
    y_ = balance_data.values[:, 7]
    y_ = y_.reshape(-1, 1)
    encoder = OneHotEncoder()
    Y = encoder.fit_transform(y_).toarray()  # Convert to dense NumPy array
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)
    return train_x, test_x, train_y, test_y

def upload(): 
    global filename
    filename = filedialog.askopenfilename(initialdir="dataset")
    if filename:
        text.delete('1.0', tk.END)
        text.insert(tk.END, f"{filename} loaded\n\n")
    else:
        messagebox.showwarning("Warning", "No file selected.")

def generateModel():
    global train_x, test_x, train_y, test_y
    data = importdata()
    train_x, test_x, train_y, test_y = splitdataset(data)
    text.insert(tk.END, f"Splitted Training Length: {len(train_x)}\n")
    text.insert(tk.END, f"Splitted Test Length: {len(test_x)}\n")

def ann():
    global model, ann_acc, train_x, train_y, test_x, test_y
    model = Sequential()
    model.add(Dense(200, input_shape=(7,), activation='relu', name='fc1'))
    model.add(Dense(200, activation='relu', name='fc2'))
    # Adjust output layer units according to your dataset
    model.add(Dense(19, activation='softmax', name='output'))  
    optimizer = Adam(lr=0.001)
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print('CNN Neural Network Model Summary: ')
    print(model.summary())
    model.fit(train_x, train_y, verbose=2, batch_size=5, epochs=200)
    results = model.evaluate(test_x, test_y)
    text.insert(tk.END, f"ANN Accuracy for dataset {filename}\n")
    text.insert(tk.END, f"Accuracy Score: {results[1]*100}\n\n")
    ann_acc = results[1] * 100

def predictFuel():
    global testdata, predictdata, model
    text.delete('1.0', tk.END)
    filename = filedialog.askopenfilename(initialdir="dataset")
    if filename:
        testdata = pd.read_csv(filename)
        testdata = testdata.values[:, 0:7]
        predictdata = np.argmax(model.predict(testdata), axis=1)
        print(predictdata)
        for i in range(len(testdata)):
            text.insert(tk.END, f"{str(testdata[i])} Average Fuel Consumption: {str(predictdata[i])}\n")
    else:
        messagebox.showwarning("Warning", "No file selected for prediction.")

def graph():
    global testdata, predictdata
    if testdata is None or predictdata is None:
        messagebox.showwarning("Warning", "No prediction data available.")
        return
    x = list(range(len(testdata)))
    plt.plot(x, predictdata)
    plt.xlabel('Vehicle ID')
    plt.ylabel('Fuel Consumption/10KM')
    plt.title('Average Fuel Consumption Graph')
    plt.show()

font = ('times', 16, 'bold')
title = Label(main, text='A Machine Learning Model for Average Fuel Consumption in Heavy Vehicles')
title.config(bg='greenyellow', fg='dodger blue')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text = Text(main, height=20, width=150)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50, y=120)
text.config(font=font1)

uploadButton = Button(main, text="Upload Heavy Vehicles Fuel Dataset", command=upload)
uploadButton.place(x=50, y=550)
uploadButton.config(font=font1)  

modelButton = Button(main, text="Read Dataset & Generate Model", command=generateModel)
modelButton.place(x=420, y=550)
modelButton.config(font=font1) 

annButton = Button(main, text="Run ANN Algorithm", command=ann)
annButton.place(x=760, y=550)
annButton.config(font=font1) 

predictButton = Button(main, text="Predict Average Fuel Consumption", command=predictFuel)
predictButton.place(x=50, y=600)
predictButton.config(font=font1) 

graphButton = Button(main, text="Fuel Consumption Graph", command=graph)
graphButton.place(x=420, y=600)
graphButton.config(font=font1) 

exitButton = Button(main, text="Exit", command=main.quit)
exitButton.place(x=760, y=600)
exitButton.config(font=font1)

main.config(bg='LightSkyBlue')
main.mainloop()
