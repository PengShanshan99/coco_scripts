# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 16:55:46 2020

@author: LETONG WEI
"""
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import opts
import yaml
import misc.utils as utils
# from sample import predict
from integrate_demo import init_model, demo_simple

def open_file():
    entry_filename.delete(0, END)
    filename = tk.filedialog.askopenfilename(title='upload image file', filetypes=[('png', '*.png')])
    entry_filename.insert('insert', filename)
    print(filename)
    img = Image.open(filename)
    tkimg = PhotoImage(file=filename).subsample(2, 2)
    display = Label(right_frame, image=tkimg, width=600, height=600).grid(row = 0, column=0, padx=5, pady=5)
    display.image = tkimg
    
def print_file():
    name = entry_filename.get()  
    print(name)    
    content = demo_simple(obj_det_model, model, name)

    text = tk.Label(left_frame, height=3, width=30, text=content)
    text.grid(row=3, column=0, padx=10, pady=5)

if __name__ == '__main__':
    opt = opts.parse_opt()
    if opt.path_opt is not None:
        with open(opt.path_opt, 'r') as handle:
            options_yaml = yaml.load(handle)
        utils.update_values(options_yaml, vars(opt))
    print(opt)
    obj_det_model, model = init_model(opt)
    
    window = tk.Tk()
    window.title("Deep Learning Project")
    window.config(bg = "skyblue")
    window.geometry("850x600")

    # create frame widget
    left_frame = Frame(window, width = 200, height = 600, bg = "grey")
    left_frame.grid(row = 0, column = 0, padx = 10, pady = 5)
    right_frame = Frame(window, width = 650, height = 600, bg = "grey")
    right_frame.grid(row = 0, column = 1, padx = 10, pady = 5)

    button_import = tk.Button(left_frame, text="Upload Image", command=open_file)
    button_import.grid(row=1, column=0, padx=10, pady=5)

    entry_filename = tk.Entry(left_frame, width=50)
    entry_filename.grid(row=0, column=0, padx=10, pady=5)

    print_button = tk.Button(left_frame, text="Predict", command=print_file)
    print_button.grid(row=2, column=0, padx=10, pady=5)
    window.mainloop()
