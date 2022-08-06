#---------Imports
from keras.models import load_model
import tkinter as tk
from PIL import ImageDraw, Image
import numpy as np
import os
import sys
#---------End of imports

### CLASSES ###
class App(tk.Tk):

    def __init__(self, model):
        tk.Tk.__init__(self)
        self.model = model
        self.x = self.y = 0

        # Creating elements
        self.image = Image.new("RGB", (300, 300), 'white') # internal image of canvas
        self.image_draw = ImageDraw.Draw(self.image)
        self.canvas = tk.Canvas(self, width=300, height=300, bg="white", cursor="cross") # canvas to draw on
        self.label = tk.Label(self, text="Thinking..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text="Recognise", command =    self.classify_handwriting)
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_all)

        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W)
        self.label.grid(row=0, column=1,pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        HWND = self.canvas.winfo_id() # get the handle of the canvas
        digit, acc = predict_digit(self.image, self.model)
        self.label.configure(text= str(digit)+', '+ str(int(acc*100))+'%')

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 8
        self.canvas.create_oval(self.x - r, self.y - r, self.x + r, self.y + r, fill='black') # show on canvas
        self.image_draw.ellipse((self.x - r, self.y - r, self.x + r, self.y + r), fill='black') # internal image

### FUNCTIONS ###
def predict_digit(img, model):

    img = img.resize((28,28)) # resize image to 28x28 pixels
    img = img.convert('L') # convert rgb to grayscale
    img = np.array(img)
    img = img.reshape(1,28,28,1) # reshapeto support our model input and normalizing
    img /= 255.0
    res = model.predict([img])[0] # predict class

    return np.argmax(res), max(res)

### MAIN FLOW ###
if __name__ == '__main__':

    model_path = 'mnist.h5'

    if not exists('mnist.h5'):
        print('Please run build_model.py first to build the classifier model')
        sys.exit()

    model = load_model(model_path)
    app = App(model)
    tk.mainloop()
