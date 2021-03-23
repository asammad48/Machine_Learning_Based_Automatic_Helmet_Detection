import tensorflow as tf
import numpy as np
from keras.preprocessing import image
import tkinter as tk
from pathlib import Path
from PIL import Image, ImageTk
import cv2
import tkinter.filedialog
from datetime import date
import numpy as np 
import time
import pyautogui

class MainWindow():
    def UploadAction(self):
        self.filename = tkinter.filedialog.askopenfilename()
        print(self.filename)
        self.cap = cv2.VideoCapture(self.filename)
        #self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        #self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        #self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.canvas = tk.Canvas(self.Video_Frame, width=934, height=600)
        self.canvas.grid(row=0, column=0)
       
        myScreenshot = pyautogui.screenshot(region=(1001,166, 365, 600))
        myScreenshot.save(r'1.png')
        self.count_bike=0
        # Update image on canvas
        self.update_image()
    def __init__(self, window):
        self.end_time=1.7
        self.start_time=0
        self.bikenumber=0
        self.carnumber=0
        self.window = window
        self.window.geometry('1300x700')
        self.interval = 20 # Interval in ms to get the latest frame
        
        self.Full_Frame = tk.Frame(self.window, background="black")
        self.Menu_Bar = tk.Frame(self.Full_Frame , background="white")
        self.Video_Frame = tk.Frame(self.Full_Frame , background="black")
        self.Receipt_Frame = tk.Frame(self.Full_Frame , background="white")
        
        
        self.Full_Frame.place(x=0, y=0, anchor="nw", width=1300, height=703)
        self.Menu_Bar.place(x=0, y=0, anchor="nw", width=1300, height=100)
        self.Video_Frame.place(x=0, y=101, anchor="nw", width=934, height=600)
        self.Receipt_Frame.place(x=935, y=101, anchor="nw", width=365, height=600)
        self.fgbg2 = cv2.createBackgroundSubtractorMOG2()
        self.kernel = np.ones((5,5),np.uint8)
        
        self.menubar()
        
        self.numberplate_image=None
        self.appearance_image=None
        self.number=''
        self.appearence=''
        self.go_to_Receipt()
        # Create canvas for image
        
    def update_image(self):
        start_frame_time = time.time () 
        # Get the latest frame and convert image format
        ret, image_org = self.cap.read()
        if ret == True: 
            dim=(934,600)
            #image_org=cv2.resize(image_org,dim,interpolation=cv2.INTER_AREA)
            frame=image_org.copy()
            binary = self.fgbg2.apply(frame)
            opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.kernel)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, self.kernel)
            x1=0
            y1=1000
            x2=3840
            y2=1600
            cv2.line(frame, (x1,y1), (x2,y2),(0, 255, 0) , 2)
            contours, hierarchy = cv2.findContours(image = closing,mode = cv2.RETR_TREE,method = cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key = cv2.contourArea, reverse = True)
            for c in contours:
                (x, y, w, h) = cv2.boundingRect(c)
                if w>160 and h>370:
                    if detect_collision(x,y,w,h,x1,y1,x2,y2)=='colision':
                        image_bike=image_org[y-100:y+h,x:x+w]
                        rect=cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 0, 255), 1)
                        image_predict=frame[y:y+h,x:x+w]
                        image_predict=cv2.resize(image_predict,(64,64),interpolation=cv2.INTER_AREA)
                        test_image=image.img_to_array(image_predict)
                        test_image=np.expand_dims(test_image,axis=0)
                        name=predict_class(test_image)
                        cv2.putText(rect, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                        self.count_bike=self.count_bike+1
                        self.numberplate_image=None
                        self.appearance_image=None
                        self.number=''
                        self.appearence=''
                        copy_of_image_bike=image_bike.copy()
                        #image_bike=cv2.resize(image_bike,(500,700),interpolation=cv2.INTER_AREA)
                        if name=='bike':
                            if self.start_time!=0:
                                self.end_time=time.time() - self.start_time
                            #cv2.imwrite(str(self.count_bike)+'.jpg',image_bike)
                            if not(image_bike.size==0):
                                data_list=predict_yolo(image_bike)
                            
                                #numberplate_image=image_bike.copy()
                                #appearance_image=image_bike.copy()
                                #cv2.imwrite(str(self.count_bike)+'.jpg',image_bike)
                                for i in range(len(data_list)):
                                    #print(data_list[i])
                                    if i%5==0:
                                        if i==0:
                                            if self.end_time>=1.6:
                                                self.bikenumber=self.bikenumber+1
                                                self.bike_value.set(self.bikenumber)
                                        if data_list[i]=='numberplate':
                                            self.numberplate_image=copy_of_image_bike[int(data_list[i+2]):int(data_list[i+4])+int(data_list[i+2]),int(data_list[i+1]):int(data_list[i+3])+int(data_list[i+1])]
                                            #cv2.imwrite('numberplate_image.jpg',numberplate_image)
                                        else:
                                            self.appearance_image=copy_of_image_bike[int(data_list[i+2]):int(data_list[i+4])+int(data_list[i+2]),int(data_list[i+1]):int(data_list[i+3])+int(data_list[i+1])]
                                            #cv2.imwrite('appearance_image.jpg',appearance_image)
                                            self.appearence=data_list[i]
                                self.go_to_Receipt()
                                self.start_time = time.time()
                        else:
                            if self.start_time!=0:
                                self.end_time=time.time() - self.start_time
                            if self.end_time>=1.6:
                                self.carnumber=self.carnumber+1
                                self.car_value.set(self.carnumber)
                            self.start_time = time.time()
                            
                    else:
                        rect=cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2)
                        image_predict=frame[y:y+h,x:x+w]
                        image_predict=cv2.resize(image_predict,(64,64),interpolation=cv2.INTER_AREA)
                        test_image=image.img_to_array(image_predict)
                        test_image=np.expand_dims(test_image,axis=0)
                        name=predict_class(test_image)
                        cv2.putText(rect, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    
        frame=cv2.resize(frame,dim,interpolation=cv2.INTER_AREA)
        self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # to RGB
        self.image = Image.fromarray(self.image) # to PIL format
        self.image = ImageTk.PhotoImage(self.image) # to ImageTk format
        # Update image
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)
        time_taken = time.time () - start_frame_time
        self.fps = 1. / time_taken 
        self.fps_value.set(self.fps)
        # Repeat every 'interval' ms
        self.window.after(self.interval, self.update_image)
    def menubar(self):
        self.img1 = ImageTk.PhotoImage(Image.open(r"numl.png"))
        panel = tk.Label(self.Menu_Bar, image = self.img1, background="white")
        panel.grid(row=0,column=0,rowspan=10)
        system_name = tk.Label(self.Menu_Bar, text="NATIONAL UNIVERSITY OF MODERN LANGUAGES", background="white",font=('Helvetica', 16, 'bold'))
        system_name.grid(row=1,column=1,rowspan=5)
        system_name1 = tk.Label(self.Menu_Bar, text="ISLAMABAD, PARKING SYSTEM", background="white",font=('Helvetica', 16, 'bold'))
        system_name1.grid(row=6,column=1,rowspan=5)
        self.Video_File_Button = tk.Button(self.Menu_Bar, text='Open Video File for Testing System',background="white",font=('Helvetica', 11, 'bold'),borderwidth=0,command=self.UploadAction)
        self.Video_File_Button.grid(row=1,column=3,rowspan=5,columnspan=2,padx=50)
        self.fps_label=tk.Label(self.Menu_Bar, text="FPS : ", background="white",font=('Helvetica', 16, 'bold'))
        self.fps_label.grid(row=6,column=3,rowspan=5)
        self.fps_value=tk.StringVar()
        self.fps_label_value=tk.Label(self.Menu_Bar,textvariable=self.fps_value, background="white",font=('Helvetica', 16, 'bold'))
        self.fps_label_value.grid(row=6,column=4,rowspan=5)
        self.bike_count=tk.Label(self.Menu_Bar,text="BIKE : ", background="white",font=('Helvetica', 16, 'bold'))
        self.bike_count.grid(row=1,column=5,rowspan=5)
        self.bike_value=tk.StringVar()
        self.bike_number=tk.Label(self.Menu_Bar,textvariable=self.bike_value, background="white",font=('Helvetica', 16, 'bold'))
        self.bike_number.grid(row=1,column=6,rowspan=5)
        
        self.car_count=tk.Label(self.Menu_Bar,text="CAR : ", background="white",font=('Helvetica', 16, 'bold'))
        self.car_count.grid(row=6,column=5,rowspan=5)
        self.car_value=tk.StringVar()
        self.car_number=tk.Label(self.Menu_Bar,textvariable=self.car_value, background="white",font=('Helvetica', 16, 'bold'))
        self.car_number.grid(row=6,column=6,rowspan=5)
        
    def go_to_Receipt(self):
        if self.appearance_image is not None:
            self.helemt_ki_image=self.appearance_image
            self.helemt_ki_image = cv2.resize(self.helemt_ki_image, (150,130), interpolation = cv2.INTER_AREA)
        else:
            self.helemt_ki_image=np.zeros([130,150,3],dtype=np.uint8)
            self.helemt_ki_image.fill(255)
        if self.numberplate_image is not None:
            self.plate_ki_image=self.numberplate_image
            print(self.plate_ki_image.shape)
            self.plate_ki_image = cv2.resize(self.plate_ki_image, (150,50), interpolation = cv2.INTER_AREA)
        else:
            self.plate_ki_image=np.zeros([50,150,3],dtype=np.uint8)
            self.plate_ki_image.fill(255)
        
        self.Generate_receipt()
        
        
    
    def Generate_receipt(self):    
        today = date.today()
        mainframe = tk.Frame(self.Receipt_Frame, background="black")
        labelframe = tk.Frame(mainframe, background="white")
        buttonframe = tk.Frame(mainframe, background="white")
        thanksframe = tk.Frame(mainframe, background="white")
        leftframe = tk.Frame(mainframe, background="white")
        rightframe = tk.Frame(mainframe, background="white")
        
        mainframe.place(x=0, y=0, anchor="nw", width=365, height=600)
        leftframe.place(x=0, y=0, anchor="nw", width=25, height=600)
        rightframe.place(x=350, y=0, anchor="nw", width=15, height=600)
        labelframe.place(x=25, y=0, anchor="nw", width=325, height=160)
        buttonframe.place(x=25, y=161, anchor="nw", width=325, height=300)
        thanksframe.place(x=25, y=462, anchor="nw", width=325, height=138)
        
        
        
        
        self.img = ImageTk.PhotoImage(Image.open(r"new.png"))
        panel = tk.Label(labelframe, image = self.img, background="white")
        panel.grid(row=0,column=0,rowspan=10)
        name1=tk.Label(labelframe,text="National University of", background="white",font=('Helvetica', 11, 'bold'))
        name2=tk.Label(labelframe,text="Modern Languages", background="white",font=('Helvetica', 11, 'bold'))
        name3=tk.Label(labelframe,text="H-9/4, Islamabad", background="white",font=('Helvetica', 11, 'bold'))
        self.var1 = tk.StringVar()
        date1=tk.Label(labelframe,textvariable=self.var1, background="white",font=('Helvetica', 11, 'bold'))
        name1.grid(row=0,column=1,rowspan=1)
        name2.grid(row=1,column=1,rowspan=1)
        name3.grid(row=2,column=1,rowspan=1)
        thankslabel=tk.Label(thanksframe,text="Thanks for using our service!", background="white",font=('Helvetica', 16, 'bold'))
        #thankslabel.config(anchor=CENTER)
        thankslabel.grid(row=0,column=1,rowspan=1,ipadx=10)
        self.var1.set(today.strftime("%B %d, %Y"))
        date1.grid(row=5,column=1,rowspan=1)
        title_receipt=tk.Label(buttonframe,text="Parking Receipt", background="white",font=('Helvetica', 16, 'bold'))
        title_receipt.grid(row=0,column=1,rowspan=1,columnspan=6,padx=65,pady=10)
        self.Helmet_var=tk.StringVar()
        helmet=tk.Label(buttonframe,textvariable=self.Helmet_var, background="white",font=('Helvetica', 16, 'bold'))
        self.Helmet_var.set(self.appearence)
        helmet.grid(row=1,column=1,columnspan=1,padx=0,pady=10)
        
        #helemt_ki_image = scipy.misc.toimage(helemt_ki_image)
        self.helmet_img = ImageTk.PhotoImage(Image.fromarray(self.helemt_ki_image))
        panel_img = tk.Label(buttonframe, image = self.helmet_img, background="white")
        panel_img.grid(row=1,column=2,rowspan=10,padx=40)
        
        
        
        self.plate_img = ImageTk.PhotoImage(Image.fromarray(self.plate_ki_image))
        panel_plate_img = tk.Label(buttonframe, image = self.plate_img, background="white")
        panel_plate_img.grid(row=12,column=2,rowspan=3,padx=40)
        self.Plate_var=tk.StringVar()
        platename=tk.Label(buttonframe,textvariable=self.Plate_var, background="white",font=('Helvetica', 16, 'bold'))
        self.Plate_var.set(self.number)
        platename.grid(row=12,column=1,columnspan=1)
        self.Price_var=tk.StringVar()
        Price_receipt=tk.Label(buttonframe,textvariable=self.Price_var, background="white",font=('Helvetica', 18, 'bold'))
        if(self.appearence=='helmet'):
            self.Price_var.set("PAID : 10 RS")
        else:
            self.Price_var.set("PAID : 20 RS")
        Price_receipt.grid(row=15,column=1,rowspan=1,columnspan=6)
    
def predict_class(test_image):
    #test_image=image. load_img('',target_size=(64,64))
    #test_image=image.img_to_array(test_image)
    #test_image=np.expand_dims(test_image,axis=0)
    #result=CNN_Classifier.predict(test_image)
    #score = tf.nn.softmax(result[0])
    result=model.predict(test_image)
    
    
    #print(result[0][0])
    if int(result[0][0])==1:
        return 'bike'
    else:
        return 'car'
        
    #print(
    #    "This image most likely belongs to {} with a {:.2f} percent confidence."
    #    .format(class_names[np.argmax(score)], 100 * np.max(score))
    #)
def detect_collision(x,y,w,h,x1,y1,x2,y2):
    linex=-(y2-y1)
    liney=x2-x1
    #print(linex)
    #print(liney)
    liney12=-(-(y1)*liney)
    linex12=-(x1)*linex
    #print(liney12)
    #print(linex12)
    lineconstant=liney12+linex12
    #print(lineconstant)
    bboxy=1
    bboxx=0
    bboxconstant=y+h
    #print(bboxconstant)
    simultaneousconstant1=lineconstant
    simultaneousconstant2=-(liney*bboxconstant)
    simultaneousconstant=simultaneousconstant1+simultaneousconstant2
    sinmultaneousx=linex
    xintersect=simultaneousconstant/sinmultaneousx
    if xintersect>=x and xintersect<=w+x and xintersect-x<=220:
        return 'colision'
        #print(xintersect)
    else:
        return 'not_colide' 
def predict_yolo(img):
    data_list=[]
    classes = []
    with open("classes.txt", "r") as f:
        classes = f.read().splitlines()
    
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(100, 3))
    
    #img = cv2.imread(image_path)
    height, width, _ = img.shape
    #print(height)
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)
    boxes = []
    confidences = []
    class_ids = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
    #print(indexes)
    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            data_list.append(label)
            data_list.append(x)
            data_list.append(y)
            data_list.append(w)
            data_list.append(h)
            
            #print(label)
            confidence = str(round(confidences[i],2))
            color = colors[i]
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)
    #cv2.imwrite('1.jpg',img)
    #cv2.imshow('image',img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return data_list
#import time

#start = time.time()
#predict_yolo(image)
#print(f'Time: {time.time() - start}')
    
    
    
    
if __name__ == "__main__":
    model = tf.keras.models.load_model('saved_model_1hour/my_model')
    net = cv2.dnn.readNet('yolov3_training_final.weights', 'yolov3_testing.cfg')
    root = tk.Tk()
    MainWindow(root)
    root.mainloop()



