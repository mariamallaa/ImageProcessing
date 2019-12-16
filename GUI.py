from tkinter import *
from tkinter import filedialog
from skimage import io 
from commonfunctions import *
import cv2
from PIL import ImageTk,Image
from filters import *


master=Tk()

master.source_file=""
Label(master, text='Choose an image').grid(row=0)
#file_name=Entry(master).grid(row=0,column=1)
master.img=[]
master.eye=IntVar()
master.mouth=IntVar()
master.nose=IntVar()
master.hat=IntVar()

def apply_filter():
    print("hi")
    
    print(master.eye.get(),master.mouth.get(),master.nose.get(),master.hat.get())
    if master.eye.get() ==1:
        master.eye.set(0)
        img=master.img.copy()
        master.destroy()
        apply(img,"eye") 
        
        #master.destroy()
    elif master.mouth.get() ==1:
        master.mouth.set(0)
        img=master.img.copy()
        master.destroy()
        apply(img,"mouth")
        
    elif master.nose.get()==1:
        master.nose.set(0)
        img=master.img.copy()
        master.destroy()
        apply(master.img,"nose")
        
    elif master.hat.get()==1:
        master.hat.set(0)
        img=master.img.copy()
        master.destroy()
        apply(master.img,"hat")
        
    
def browse_file():
    
    source_file =  filedialog.askopenfilename(parent=master, initialdir= "/", title='Please select a file',filetype=[("image files","*.jpeg;*.png;*.jpg")])   
    master.img=io.imread(source_file)
    Label(master, text='Choose a filter ').grid(row=3) 
    Checkbutton(master, text='Glasses',variable=master.eye,command=apply_filter).grid(row=4, sticky=W)  
    Checkbutton(master, text='Clown Nose',variable=master.nose, command=apply_filter).grid(row=5, sticky=W) 
    Checkbutton(master, text='Funny Mouth', variable=master.mouth,command=apply_filter).grid(row=6, sticky=W)
    Checkbutton(master, text='Hat',variable=master.hat, command=apply_filter).grid(row=7, sticky=W)

def take_photo():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("take a photo")
    ret, frame = cam.read()
    cv2.imshow("take a photo", frame)
    img =frame
    master.img = img[:, :, ::-1]
    
    Label(master, text='Choose a filter ').grid(row=3) 
    Checkbutton(master, text='Glasses',variable=master.eye,command=apply_filter).grid(row=4, sticky=W)  
    Checkbutton(master, text='Clown Nose',variable=master.nose, command=apply_filter).grid(row=5, sticky=W) 
    Checkbutton(master, text='Funny Mouth', variable=master.mouth,command=apply_filter).grid(row=6, sticky=W)
    Checkbutton(master, text='Hat',variable=master.hat, command=apply_filter).grid(row=7, sticky=W)

browse=Button(master, text='Browse', width=15, command=browse_file) 
browse.grid(row=1,column=0)
#Label(master, text='OR').grid(row=1,column=1)
take_photo=Button(master, text='Take a photo', width=15, command=take_photo) 
take_photo.grid(row=1,column=1)
master.mainloop()

