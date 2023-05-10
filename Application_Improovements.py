##  python -m PyInstaller --name MegaMold_Time_Management_Application --icon ..\mmi.ico ..\Application_Improovements.py


# coll = COLLECT(
#     exe, Tree('C:\\Users\\Research\\MegaMold Application\\Deployment_19042023\\'),
#     a.binaries,
#     a.zipfiles,
#     a.datas,
#     strip=False,
#     upx=True,
#     upx_exclude=[],
#     name='MegaMold_Time_Management_Application',
# )

## from kivy_deps import sdl2, glew


from kivymd.uix.label import MDLabel
from kivymd.uix.screen import MDScreen
from kivy.lang.builder import Builder
from kivy.factory import Factory as Factory
from kivymd.app import MDApp
from kivymd.uix.button import MDRectangleFlatIconButton
from kivymd.icon_definitions import md_icons
import sys
import sklearn
from kivy.uix.screenmanager import ScreenManager, Screen
import math
from kivy.clock import Clock
from kivy.graphics import Color, Line
import os
import time
import cv2
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics.texture import Texture
import numpy as np
import face_recognition
from datetime import datetime
import datetime
import pandas as pd
from imutils.video import VideoStream
from keras.utils import img_to_array
from keras.models import load_model
import pickle
import imutils
from kivy.core.window import Window
from kivymd.uix.dialog import MDDialog
from kivy.metrics import dp
from kivymd.uix.button import MDFlatButton
import xml.etree.ElementTree as gfg
from kivymd.uix.spinner import MDSpinner
from kivy.properties import StringProperty
from kivy.uix.spinner import Spinner, SpinnerOption
from kivy.metrics import dp
from kivy.properties import ListProperty
from kivymd.uix.textfield import MDTextField
from kivy.properties import NumericProperty
import threading
from kivy.graphics import Rectangle, Color
import pyodbc

os.environ['KIVY_METRICS_DENSITY'] = '1'
os.environ['KIVY_METRICS_FONTSCALE'] = '1'

def get_employee_id(employee_name, cursor):
    query = "SELECT EmployeeID FROM EmployeeInformation WHERE EmployeeName=?"
    cursor.execute(query, (employee_name,))
    result = cursor.fetchone()

    if result:
        return result[0]
    else:
        return None


def create_connection():
    server = '.\\SQLEXPRESS'
    database = 'MegaMoldInternationalDB'
    driver = '{ODBC Driver 17 for SQL Server}'
    connection_string = f'DRIVER={driver};SERVER={server};DATABASE={database};Trusted_Connection=yes;'
    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()
    print(" Connected to Database")
    return conn, cursor


def insert_employee_Registration(employee_id, employee_name):
    conn, cursor = create_connection()
    query = "INSERT INTO EmployeeInformation (TimeStamp, EmployeeID, EmployeeName) VALUES (?, ?, ?)"
    current_timestamp = datetime.datetime.now()
    cursor.execute(query, (current_timestamp, employee_id, employee_name))
    conn.commit()
    cursor.close()
    conn.close()


def insert_employee_Login(employee_name):
    conn, cursor = create_connection()

    # Fetch the EmployeeID from the EmployeeInformation table
    employee_id = get_employee_id(employee_name, cursor)

    if employee_id:
        query = "INSERT INTO EmployeeActivity (EmployeeID, EmployeeName, EmployeeActivity, Timestamp) VALUES (?, ?, ?, ?)"
        current_timestamp = datetime.datetime.now()
        cursor.execute(query, (employee_id, employee_name, "LOGIN", current_timestamp))
        conn.commit()

    cursor.close()
    conn.close()

def insert_employee_ClockIn(employee_name):
    conn, cursor = create_connection()

    # Fetch the EmployeeID from the EmployeeInformation table
    employee_id = get_employee_id(employee_name, cursor)

    if employee_id:
        query = "INSERT INTO EmployeeActivity (EmployeeID, EmployeeName, EmployeeActivity, Timestamp) VALUES (?, ?, ?, ?)"
        current_timestamp = datetime.datetime.now()
        cursor.execute(query, (employee_id, employee_name, "CLOCK IN", current_timestamp))
        conn.commit()

    cursor.close()
    conn.close()

def insert_employee_ClockOut(employee_name):
    conn, cursor = create_connection()

    # Fetch the EmployeeID from the EmployeeInformation table
    employee_id = get_employee_id(employee_name, cursor)

    if employee_id:
        query = "INSERT INTO EmployeeActivity (EmployeeID, EmployeeName, EmployeeActivity, Timestamp) VALUES (?, ?, ?, ?)"
        current_timestamp = datetime.datetime.now()
        cursor.execute(query, (employee_id, employee_name, "CLOCK OUT", current_timestamp))
        conn.commit()

    cursor.close()
    conn.close()

def insert_employee_EnterTime(employee_name):
    conn, cursor = create_connection()

    # Fetch the EmployeeID from the EmployeeInformation table
    employee_id = get_employee_id(employee_name, cursor)

    if employee_id:
        query = "INSERT INTO EmployeeActivity (EmployeeID, EmployeeName, EmployeeActivity, Timestamp) VALUES (?, ?, ?, ?)"
        current_timestamp = datetime.datetime.now()
        cursor.execute(query, (employee_id, employee_name, "JOB ENTRY", current_timestamp))
        conn.commit()

    cursor.close()
    conn.close()



def read_employee_data(filename):
    df = pd.read_excel(filename, engine='openpyxl')
    if 'EmployeeName' not in df.columns:
        raise ValueError("The Excel file must contain an 'EmployeeName' column.")
    allowed_operations_columns = [col for col in df.columns if col.startswith('Allowedoperations')]
    if not allowed_operations_columns:
        raise ValueError("The Excel file must contain at least one 'Allowedoperations' column.")
    # Create a dictionary with EmployeeName as keys and AllowedOperations as values
    employee_data = {}
    for _, row in df.iterrows():
        employee_name = row['EmployeeName']
        allowed_operations = [row[col] for col in allowed_operations_columns if not pd.isna(row[col])]
        employee_data[employee_name] = allowed_operations
    return employee_data

filename = 'OperationsManagement.xlsx'
employee_data = read_employee_data(filename)
print(employee_data)

prototxt = './deploy.prototxt'
caffemodel = './res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
model = load_model('liveness.model')
le = pickle.loads(open('le.pickle', "rb").read())

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

class SM(ScreenManager):
    pass

class CustomMDTextField(MDTextField):
    max_length = NumericProperty(0)

    def insert_text(self, substring, from_undo=False):
        if self.max_length and (len(self.text) + len(substring) > self.max_length):
            substring = substring[: self.max_length - len(self.text)]
        filtered_text = ''.join([c for c in substring if c.isalpha() or c.isspace()])
        return super(CustomMDTextField, self).insert_text(filtered_text, from_undo)



class CustomEmpIDTextField(MDTextField):
    max_length = NumericProperty(0)

    def insert_text(self, substring, from_undo=False):
        if self.max_length and (len(self.text) + len(substring) > self.max_length):
            substring = substring[: self.max_length - len(self.text)]
        filtered_text = ''.join([c for c in substring if c.isdigit()])
        return super(CustomEmpIDTextField, self).insert_text(filtered_text, from_undo)


class HomeScreen(MDScreen):
    pass
class RegistrationScreen(MDScreen):
    def on_enter(self, *args):
        self.ids.empname.text = ""
        self.ids.empid.text = ""
        self.ids.reg_but.enabled = False

    def update_button_state(self):
        empname = self.ids.empname.text
        empid = self.ids.empid.text

        if empname and empid:
            self.ids.reg_but.disabled = False
        else:
            self.ids.reg_but.disabled = True

class FaceEncodingInputScreen(MDScreen):

    def reset_texture(self):
        self.ids.video_FIS.texture = None

    def on_pre_enter(self):
        self.reset_texture()


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = None
        self.image = Image()
        self.success_dialog = None

    def on_enter(self, *args):
        self.ids.FIS_but.bind(on_press=self.callback)
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        Clock.schedule_interval(self.update, 1.0 / 15.0)
        return self.image

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            buffer = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
            self.ids.video_FIS.texture = texture

    def callback(self, instance):
        global EN
        global EID
        select_screen = self.manager.get_screen("reg")
        EID = select_screen.ids.empid.text
        EN = select_screen.ids.empname.text
        insert_employee_Registration( EID, EN)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        ret, frame = self.cap.read()
        if ret:
            cv2.imwrite("Face_Directory/{}.png".format(EN), frame)
            self.cap.release()
            Clock.unschedule(self.update)
            self.success_dialog = MDDialog(title="Registration Successful",
                                           text="Employee has been successfully registered.",
                                           size_hint=(.8, None), height=dp(200),
                                           buttons=[MDFlatButton(text="OK", on_release=self.change_screen)])
            self.success_dialog.open()
            sm = MDApp.get_running_app().root
            FR_Ref = sm.get_screen("FR")
            FR_Ref.Encoding_Reload_Prompt = True
            # my_list = [EID, EN, " "]
            # my_string = ','.join(my_list)
            # # IPL_Service.service.User_Master('SP_Insert_User_Master', my_string)
        else:
            print("Failed to capture image")

    def change_screen(self, instance):
        self.manager.current = "home"
        self.success_dialog.dismiss()
        Clock.unschedule(self.update)
        self.capture.release()
        self.cap.release()
        cv2.destroyAllWindows()

    def on_stop(self):
        self.capture.release()
        Clock.unschedule(self.update)
        cv2.destroyAllWindows()

    def on_leave(self):
        cv2.destroyAllWindows()


class LoginScreen(MDScreen):

    def show_login_success_dialog(self):
        self.Login_Success_dialog = MDDialog(title="Welcome " + self.name_db,
                                             text=" Please Press the Proceed Button to Continue to Time Entries ",
                                             size_hint=(.8, None), height=dp(300),
                                             buttons=[MDFlatButton(text=" Proceed ",
                                                                   on_release=self.change_screen_Login_Success_dialog)])
        # self.manager.current = "OP"
        self.Login_Success_dialog.open()


    def show_spoof_dialog(self):
        self.spoof_dialog = MDDialog(title="SPOOF ATTACK ALERT!!!",
                                     text="Please Do not try to spoof the system. This instance has been recorded",
                                     size_hint=(.8, None), height=dp(300),
                                     buttons=[MDFlatButton(text="Close",
                                                           on_release=self.change_screen_Spoof)])
        self.spoof_dialog.open()

    def reset_video_texture(self):
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        buffer = cv2.flip(blank_frame, 0).tobytes()
        texture = Texture.create(size=(blank_frame.shape[1], blank_frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.ids.video.texture = texture


    def Go_Home(self, *args):
        self.manager.current = "home"
        self.capture.release()
        Clock.unschedule(self.update)
        Clock.unschedule(self.Go_Home)
        self.on_stop()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name_db = ''
        self.frame_count = 0
        self.classNames = []
        self.encodeListKnown = []
        self.load_images()
        self.capture = None
        self.Encoding_Reload_Prompt = False
        self.spoof_dialog = None
        self.Login_Success_dialog = None
        self.image = Image()
        self.face_detection_running = True

    def load_images(self):
        path = './Face_Directory'
        self.classNames = []
        self.encodeListKnown = []
        images = []
        myList = os.listdir(path)
        for cl in myList:
            curImg = cv2.imread(f'{path}/{cl}')
            images.append(curImg)
            self.classNames.append(os.path.splitext(cl)[0])
        self.encodeListKnown = findEncodings(images)



    def on_pre_enter(self):
        self.reset_video_texture()

    def on_enter(self, *args):

        self.face_detection_running = True  # Set the flag to True when entering the screen
        self.face_detection_thread = threading.Thread(target=self.face_detection)
        self.face_detection_thread.start()


        # global Encoding_Reload_Prompt
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        Clock.schedule_interval(self.update, 1.0 / 15.0)
        Clock.schedule_interval(self.Go_Home, 20)

        if self.Encoding_Reload_Prompt == True:
            self.load_images()
            self.Encoding_Reload_Prompt = False

        # Create a separate thread for face detection
        self.face_detection_thread = threading.Thread(target=self.face_detection)
        self.face_detection_thread.start()

        return self.image



    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            buffer = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
            self.ids.video.texture = texture

    def face_detection(self):
        while self.capture.isOpened() and self.face_detection_running:
            ret, frame = self.capture.read()
            if not ret:
                break

            # Face detection code
            frame = imutils.resize(frame, width=600)
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                         (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.7:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(w, endX)
                    endY = min(h, endY)

                    face = frame[startY:endY, startX:endX]
                    face = cv2.resize(face, (32, 32))
                    face = face.astype("float") / 255.0
                    face = img_to_array(face)
                    face = np.expand_dims(face, axis=0)

                    preds = model.predict(face)[0]
                    j = np.argmax(preds)
                    label = le.classes_[j]
                    if label == "real":
                        facesCurFrame = face_recognition.face_locations(frame)
                        encodesCurFrame = face_recognition.face_encodings(frame, facesCurFrame)
                        rects = net.forward()
                        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                            matches = face_recognition.compare_faces(self.encodeListKnown, encodeFace, tolerance=0.4)
                            faceDis = face_recognition.face_distance(self.encodeListKnown, encodeFace)
                            matchIndex = np.argmin(faceDis)

                            if matches[matchIndex]:
                                name = self.classNames[matchIndex]
                                self.name_db = name
                                self.manager.get_screen('ET').employee_name = self.name_db
                                # self.manager.get_screen('OP').name = self.name_db
                                print(self.name_db)
                                # self.Login_Success_dialog.open()
                                Clock.schedule_once(lambda dt: self.show_login_success_dialog(), 0)
                                # self.manager.current = "OP"
                                self.on_stop()
                                global name_db
                                name_db = name

                    else:
                        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        file_name = "Spoof_Attacks/" + f"{current_time}.png"
                        if not os.path.exists("./Spoof_Attacks"):
                            os.makedirs("./Spoof_Attacks")
                        cv2.imwrite(file_name, frame)
                        Clock.schedule_once(lambda dt: self.show_spoof_dialog(), 0)
                        self.manager.current = "home"
                        Clock.unschedule(self.update)
                        Clock.unschedule(self.Go_Home)
                        self.capture.release()
                        cv2.destroyAllWindows()

    def change_screen_Spoof(self, instance):
        self.spoof_dialog.dismiss()

    def change_screen_Login_Success_dialog(self, instance):
        sm = MDApp.get_running_app().root
        OP_screen = sm.get_screen("OP")
        OP_screen.db_name = self.name_db  # Set the db_name attribute instead of changing the name
        print(OP_screen.db_name)
        insert_employee_Login(self.name_db)
        self.manager.current = "OP"
        self.Login_Success_dialog.dismiss()
        self.capture.release()
        Clock.unschedule(self.update)
        Clock.unschedule(self.Go_Home)

    def on_stop(self):
        self.capture.release()
        Clock.unschedule(self.update)
        Clock.unschedule(self.Go_Home)
        if hasattr(self, "face_detection_thread"):
            self.face_detection_thread.join()

    def on_leave(self):
        self.capture.release()
        Clock.unschedule(self.update)
        Clock.unschedule(self.Go_Home)
        sm = MDApp.get_running_app().root
        enter_time_screen = sm.get_screen("ET")
        enter_time_screen.employee_name = self.name_db
        print(enter_time_screen.employee_name)
        self.face_detection_running = False
        if hasattr(self, "face_detection_thread"):
            self.face_detection_thread.join()

class OperationsScreen(MDScreen):

    db_name = StringProperty(None)

    # SM.manager.get_screen('FR').name_db = name

    def db_prompt(self):
        # enter_time_screen = SM.get_screen("ET")
        # enter_time_screen.employee_name = self.name
        print(self.db_name)
        insert_employee_EnterTime(self.db_name)

    def on_pre_enter(self, *args):
        self.ids.bottom_navigation.switch_tab("HOME")



    dialog_label_clock_in = None
    dialog_label_clock_out = None
    dialog = None

    def show_dialog_box_clock_in(self):
        Clock.schedule_once(self.create_dialog_box_clock_in, 0)

    def create_dialog_box_clock_in(self, dt):
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        self.dialog = MDDialog(  # Store dialog instance as attribute
            title="Clock In",
            type="custom",
            size_hint=(.8, None), height=dp(300),
            content_cls=MDLabel(
                text=f"Are you sure you want to clock in at {current_time}?",
                theme_text_color="Custom",
                text_color=(0, 0, 0, 1),
                font_style='H6'
            ),
            buttons=[
                MDFlatButton(
                    text="Cancel", font_style='H6', on_release=lambda x: self.dialog.dismiss(),
                ),
                MDFlatButton(            
                    self.on_leave_CI(),text="Clock In", font_style='H6', on_release=lambda x: self.clock_in()
                    
                 
                ),
            ],
        )
        self.dialog.open()

    def show_dialog_box_clock_out(self):
        Clock.schedule_once(self.create_dialog_box_clock_out, 0)

    def create_dialog_box_clock_out(self, dt):
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        self.dialog = MDDialog(
            title="Clock Out",
            type="custom",
            size_hint=(.8, None), height=dp(300),
            content_cls=MDLabel(
                text=f"Are you sure you want to clock out at {current_time}?",
                theme_text_color="Custom",
                text_color=(0, 0, 0, 1),
                font_style='H6'
            ),
            buttons=[
                MDFlatButton(
                    text="Cancel",font_style='H6', on_release=lambda x: self.dialog.dismiss()
                ),
                MDFlatButton(
                    self.on_leave_CO(),text="Clock Out", font_style='H6', on_release=lambda x: self.clock_out()
                ),
            ],
        )
        self.dialog.open()

    def clock_in(self):
        insert_employee_ClockIn(self.db_name)
        self.dialog.dismiss()
        self.manager.current = "last"

    def clock_out(self):
        insert_employee_ClockOut(self.db_name)
        self.dialog.dismiss()
        self.manager.current = "last"

    def on_leave_CI(self, *args):
        pass

        # Var=name_db
        # my_list = [Var,'1']
        # my_string = ','.join(my_list)
        # IPL_Service.service.Clock_IN_Out('SP_Insert_CLock_IN_OUT', my_string)

    def on_leave_CO(self, *args):
        pass
        # Var=name_db
        # my_list = [Var,'0']
        # my_string = ','.join(my_list)
        # IPL_Service.service.Clock_IN_Out('SP_Insert_CLock_IN_OUT', my_string)


  
  

# def GenerateXML(self,fileName) :
#
#
#
#     global root
#     root = gfg.Element("NewDataSet")
#
#     for i in range(11-1):
#
#         job='job_num'+str(i+1)
#         Operation_No='spinner_id'+str(i+1)
#         Time_Work='time_work'+str(i+1)
#
#         m1 = gfg.Element("Table")
#         root.append(m1)
#
#         b1 = gfg.SubElement(m1, "Job_Num")
#         b1.text = self.ids[str(job)].text
#
#         b2 = gfg.SubElement(m1, "Operation_No")
#         b2.text = self.ids[str(Operation_No)].text
#
#         c1 = gfg.SubElement(m1, "Time_Work")
#         c1.text = self.ids[str(Time_Work)].text
#
#         tree = gfg.ElementTree(root)
#         with open (fileName, "wb") as files :
#             tree.write(files)
#
#     global XML
#     XML=gfg.tostring(root).decode()
#
#     return XML
          

class EnterTimeScreen(MDScreen, BoxLayout):
    employee_name = StringProperty(None)
    print(employee_name)


    def on_back_button(self):
        self.manager.current = "OP"


    def __init__(self, **kwargs):
        super(EnterTimeScreen, self).__init__(**kwargs)
        print(self.employee_name)


    def on_spinner_select(self, text):
        print(f"Selected: {text}")

    def set_employee_operations(self):
        print(employee_data)
        available_operations = employee_data.get(self.employee_name, [])
        print(available_operations)
        self.ids.spinner_id1.values = available_operations
        self.ids.spinner_id2.values = available_operations
        self.ids.spinner_id3.values = available_operations
        self.ids.spinner_id4.values = available_operations
        self.ids.spinner_id5.values = available_operations
        self.ids.spinner_id6.values = available_operations
        self.ids.spinner_id7.values = available_operations
        self.ids.spinner_id8.values = available_operations
        self.ids.spinner_id9.values = available_operations
        self.ids.spinner_id10.values = available_operations
        self.ids.spinner_id11.values = available_operations

        print("Executed")

    def on_pre_enter(self):
        for i in range(1, 12):
            self.ids['job_num{}'.format(i)].text = ''
            self.ids['spinner_id{}'.format(i)].text = "No Operation Selected"
            self.ids['time_work{}'.format(i)].text = ''
            self.ids['ot{}'.format(i)].text = ''

    def on_enter(self):
        self.set_employee_operations()


    def ET(self):
        Job_No=self.ids.job_num.text
        OP_No=self.ids.spinner_id.text
        Time_Work=self.ids.result_label.text
        Var=self.employee_name
        print(Job_No,OP_No,Time_Work)
        Para = Job_No,OP_No,Time_Work
        print(Para)
        my_list = [Job_No, OP_No, Time_Work,Var]
        my_string = ','.join(my_list)
        # IPL_Service.service.Insert('SP_Insert_CLock_Master', my_string)

    def calculate_sum(self, *args):
        num_list = []
        otnum_list = []
        for i in range(1, 11):
            if f"time_work{i}" in self.ids:
                num_text = self.ids[f"time_work{i}"].text
                num = float(num_text) if num_text else 0.0
                num_list.append(num)

        for i in range(1, 11):
            if f"ot{i}" in self.ids:
                otnum_text = self.ids[f"ot{i}"].text
                otnum = float(otnum_text) if otnum_text else 0.0
                otnum_list.append(otnum)
        resultreg = sum(num_list)
        resultot = sum(otnum_list)
        rounded_resultreg = round(resultreg, 2)
        rounded_resultot = round(resultot, 2)
        self.ids.result_label_reg.text = str(rounded_resultreg)
        self.ids.result_label_ot.text = str(rounded_resultot)

        if rounded_resultreg > 8 or rounded_resultot > 8:
            # Show the MDDialog
            too_many_hours_dialog = MDDialog(
                title="Too Many Hours",
                text="You have entered more than 8 hours. Please adjust your entries.",
                size_hint=(0.8, None),
                height=300,
                buttons=[
                    MDFlatButton(
                        text="OK",
                        on_release=lambda x: too_many_hours_dialog.dismiss()
                    )
                ],
            )
            too_many_hours_dialog.open()

            # Disable the "Add Work Entry" button
            add_work_entry_button = self.ids.et_but
            add_work_entry_button.disabled = True
        else:
            # Enable the "Add Work Entry" button if it was previously disabled
            add_work_entry_button = self.ids.et_but
            add_work_entry_button.disabled = False

    def Enter_Time(self):
        if self.ids.et_but.disabled:
            return


        self.dialog = MDDialog(text="Confirm to submit yor Time Entries",
                                         size_hint=(.8, None), height=dp(300),
                                         buttons=[MDFlatButton(text="Cancel",font_style='H6', on_release=lambda x: self.dialog.dismiss()),
                                                  MDFlatButton(text=" Proceed ",font_style='H6',
                                                               on_release=lambda x: self.change_last())
                                          ])
       
       
        
        # XML_NEW=GenerateXML(self,"Catalog.xml")
        # Var=self.employee_name
        # my_list = [Var,XML_NEW]
        # my_string = ','.join(my_list)
        
       
        self.dialog.open()

    def change_last(self):
        self.save_employee_job_entries()
        self.dialog.dismiss()
        self.manager.current = "last"

    def save_employee_job_entries(self):
        conn, cursor = create_connection()
        employee_id = get_employee_id(self.employee_name, cursor)
        timestamp = datetime.datetime.now()

        for i in range(1, 12):
            job_number = self.ids[f'job_num{i}'].text
            operation_name = self.ids[f'spinner_id{i}'].text
            regular_hours = self.ids[f'time_work{i}'].text
            overtime_hours = self.ids[f'ot{i}'].text or '0'

            if not all([job_number, operation_name, regular_hours]):
                continue

            query = """INSERT INTO EmployeeJobEntries (EmployeeID, EmployeeName, Timestamp, JobNumber, 
                    OperationName, RegularHours, OvertimeHours) VALUES (?, ?, ?, ?, ?, ?, ?)"""
            try:
                cursor.execute(query, (employee_id, self.employee_name, timestamp, job_number,
                                       operation_name, regular_hours, overtime_hours))
                conn.commit()
            except Exception as e:
                print(f"Error while inserting entry: {e}")
                continue

        cursor.close()
        conn.close()

class LastScreen(Screen):
    def on_enter(self):
        Clock.schedule_once(self.timeout, 3)

    def timeout(self, dt):
        self.manager.current = 'home'

    def on_stop(self):
        pass

class ContentNavigationDrawer(BoxLayout):
    pass
class IncrediblyCrudeClock(MDLabel):
    def __init__(self, **kwargs):
        super(IncrediblyCrudeClock, self).__init__(**kwargs)
        Clock.schedule_interval(self.update, 1)

    def update(self, *args):
        now = datetime.datetime.now()
        date_str = now.strftime("%A, %B %d, %Y")
        time_str = now.strftime("%I:%M:%S %p")
        self.text = f"{date_str}\n{time_str}"


class MMI(MDApp):
    def build(self):
        Window.set_title("Mega Mold International Application")
        Window.icon = "mmi.ico"
        Window.unbind(on_request_close=self.stop)
        Window.borderless = True
        Window.fullscreen = True
        return Builder.load_file("MMI.kv")

if __name__ == "__main__":
    MMI().run()