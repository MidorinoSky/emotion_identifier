from ctypes import windll
from ttkbootstrap import Style
from ttkbootstrap import Label
from ttkbootstrap import Menu
from PIL.ImageTk import PhotoImage
from tkinter.filedialog import askopenfilename
from keras.models import model_from_json
from PIL.Image import fromarray
import numpy as np
import cv2

img_test = None
windll.shcore.SetProcessDpiAwareness(1)
capture = cv2.VideoCapture(0)
width = 800
height = 600

with open("model.json", "r") as jf:
    loaded_model_json = jf.read()
model = model_from_json(loaded_model_json)
model.load_weights("weights.h5")

emotions = {0: "angry",
            1: "disgust",
            2: "fear",
            3: "happy",
            4: "sad",
            5: "surprise",
            6: "neutral"}


def window_setup():
    global top_label, pic_label
    root = Style(theme="minty").master
    top_label = Label(root)
    top_label.pack(pady=15)
    pic_label = Label(root, image="")
    pic_label.pack()

    mainmenu = Menu(root)
    menuFile = Menu(mainmenu)
    mainmenu.add_cascade(label="菜单", menu=menuFile)
    menuFile.add_command(label="拍照", command=camera)
    menuFile.add_command(label="图片", command=browser)
    menuFile.add_separator()
    menuFile.add_command(label="退出", command=root.destroy)

    root.config(menu=mainmenu)
    return root


def identify(pic_file):
    img = cv2.imread(pic_file, 0)
    img_test = cv2.resize(img, (48, 48), interpolation=cv2.INTER_NEAREST)
    img_test = np.array([img_test]) / 255.0
    result = model.predict(img_test)
    predict = np.argmax(result)
    img = cv2.imread(pic_file)
    bl = 0.8 * min(width / img.shape[1], height / img.shape[0])
    img = cv2.resize(img, (int(bl * img.shape[1]), int(bl * img.shape[0])), interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = fromarray(img)
    return emotions[predict], img


def browser():
    file = askopenfilename()
    if file != "":
        global img_test
        result, img_test = identify(file)
        img_test = PhotoImage(img_test)
        top_label.config(text=result)
        pic_label.config(image=img_test)
    else:
        top_label.config(text="您没有选择任何文件")
        img_test = None
        pic_label.config(image=img_test)


def camera():
    global capture
    while True:
        ret, frame = capture.read()
        frame = cv2.flip(frame, 1)  # 摄像头是和人对立的，将图像左右调换回来正常显示。
        cv2.imshow("拍照识别表情", frame)
        c = cv2.waitKey(1)
        if c == 13:  # 按下enter，截图保存并退出
            cv2.imwrite("test.png", frame)
            result, img_test = identify("test.png")
            img_test = PhotoImage(img_test)
            top_label.config(text=result)
            pic_label.config(image=img_test)
            break
        elif c == 27:
            top_label.config(text="您没有拍摄任何照片")
            img_test = None
            pic_label.config(image=img_test)
            break
    cv2.destroyAllWindows()
    capture = cv2.VideoCapture(0)


if __name__ == "__main__":
    window = window_setup()
    window.title("表情识别")
    window.geometry(str(width) + "x" + str(height))
    window.resizable(False, False)
    window.mainloop()
    cv2.destroyAllWindows()

