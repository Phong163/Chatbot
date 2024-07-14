
import torch
import sys
import json
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QMovie
from PyQt5 import QtWidgets
from interface import Ui_MainWindow
import speech_recognition as sr
from function import *
from config import get_config, get_weights_file_path, get_gif
from transformer.train import get_model, get_ds,greedy_decode

#load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = get_config()
train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

# Load the pretrained weights
model_filename = get_weights_file_path(config, f"62")
state = torch.load(model_filename, map_location = device)
model.load_state_dict(state['model_state_dict'])
##
t=0
with open('data2.json','r', encoding="utf-8") as file:
    datajson = json.load(file)

def run_predict(model,user_input, tokenizer_src, tokenizer_tgt, max_len, device, print_msg):
    model.eval()
    predicted = []
    sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64).to(device)
    eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64).to(device)
    pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64).to(device)
    
    src_text = user_input
    src_text = src_text[:max_len]
    enc_input_tokens = tokenizer_src.encode(src_text).ids

    # Add sos, eos and padding to each sentence
    enc_num_padding_tokens = max_len - len(enc_input_tokens) - 2  # We will add <s> and </s>
    # We will only add <s>, and </s> only on the label
    encoder_input = torch.cat(
        [
            sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64).to(device),
            eos_token,
            torch.tensor([pad_token] * enc_num_padding_tokens, dtype=torch.int64).to(device),
        ],
        dim=0,
    )
    # Add only <s> token
    encoder_mask = (encoder_input != pad_token).unsqueeze(0).unsqueeze(0).int().to(device) # (1, 1, seq_len)
    
    model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
    
    model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
# Join tokens back into a string
    predicted.append(model_out_text)
    print_msg(f"{f'SOURCE: ':>12}{src_text}")
    print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")
    return model_out_text

def start_translate(lang):
    if is_internet_available():
            recognizer = sr.Recognizer()
            with sr.Microphone() as source:
                print("Đang nghe...")
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source)
            try:
                user_input = recognizer.recognize_google(audio, language="vi-VN")
                user_input=user_input.lower()
                print("Đã nhận được: " + user_input)
            except sr.UnknownValueError:
                print('lỗi rồi')
                user_input='xin lỗi tôi không nghe rõ, hay nói lại câu lệnh'
            user_input=translator(user_input,lang)
            play_wav(user_input,lang,speed_factor=1.1)

class MYUI(Ui_MainWindow):  
    def __init__(self):
        super().__init__()
        self.setupUi(MainWindow)
        self.pushButton_2.clicked.connect(self.start_conversation)
        self.pushButton_1.clicked.connect(self.EndWindown)
        self.pushButton_3.clicked.connect(lambda: self.changePage(2))
        self.pushButton_6.clicked.connect(lambda: self.changePage(1))
        self.start_animation()
        self.thread1 = None
        self.thread2 = None
    def start_animation(self):
        gif_path = get_gif("1_1.gif")
        self.movie = QMovie(gif_path)
        self.label.setMovie(self.movie)
        self.movie.start()
    def start_conversation(self):
        self.pushButton_2.setStyleSheet("background-color: rgb(0, 170, 255);")
        global t
        t=t+1
        if t==1:
            if not self.thread1 or not self.thread1.isRunning():
                self.thread1 = Task1Thread()
                self.thread1.update_signal.connect(self.change_gif)
                self.thread1.start()
        else:
            print('thread1 dang chay')
    def change_gif(self, gif_path):
        self.movie.stop()
        self.movie.setFileName(gif_path)
        self.movie.start()
    def chan_gif_page2(self,gif_path,index):
        movie = QMovie(get_gif("Spinner.gif"))
        if index==5:
            self.label_5.setMovie(movie)
            self.label_5.setScaledContents(True)
        if index==6:
            self.label_6.setMovie(movie)
            self.label_6.setScaledContents(True)
        movie.start()
    def changePage(self, index):
        if index==1:
            self.stackedWidget.setCurrentWidget(self.page_1)
        elif index==2:
            self.stackedWidget.setCurrentWidget(self.page_2)
    def EndWindown(self):
        app.quit()

class Task1Thread(QThread):
    update_signal = pyqtSignal(str)  # Tạo một tín hiệu để gửi đường dẫn hình ảnh mới
    def run(self):
        global t
        exc=0
        while True:
            if is_internet_available():
                recognizer = sr.Recognizer()
                with sr.Microphone() as source:
                    print("Đang nghe...")
                    self.update_signal.emit(get_gif('1_2.gif'))
                    recognizer.adjust_for_ambient_noise(source)
                    audio = recognizer.listen(source)
                try:
                    user_input = recognizer.recognize_google(audio, language="vi-VN")
                    user_input=user_input.lower()
                    print("Đã nhận được: " + user_input)
                except sr.UnknownValueError:
                    user_input = 'None'
                    exc += 1
                    if exc == 3:
                        answer = 'tạm biệt'
                        self.update_signal.emit(get_gif('1_1.gif'))
                        play_wav(answer,'vi',speed_factor=1.3)
                        break
                if user_input != 'None':
                    type=classify_input(user_input)
                    print('type',type)
                    if type==1:
                        answer,next_act=command_type1(user_input)
                        if next_act==1:
                            self.update_signal.emit(get_gif('1_4.gif'))
                            play_nhac()
                        if next_act==0:
                            break
                    elif type==2:
                        answer,lang=command_type2(user_input)
                        print('lang',lang)
                        self.update_signal.emit(get_gif('1_3.gif'))
                        play_wav(answer,'vi',speed_factor=1.3)
                        start_translate(lang)
                    elif type==3:
                        answer=command_type3(user_input)
                        self.update_signal.emit(get_gif('1_3.gif'))
                        play_wav(answer,'vi',speed_factor=1.3)
                        save_conversation(user_input,answer)
                    elif type==4:
                        max_matched_words,best_answer,question_in_data= compare(user_input, datajson)
                        if max_matched_words >0.60*len(question_in_data.split()):
                            self.update_signal.emit(get_gif('1_3.gif'))
                            print("Trả lời:", best_answer)
                            play_wav(best_answer,'vi',speed_factor=1.3)
                            repeat(best_answer)
                        else:
                            best_answer=run_predict(model,user_input, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: print(msg))
                            self.update_signal.emit(get_gif('1_3.gif'))
                            print("Trả lời:", best_answer)
                            play_wav(best_answer, 'vi', speed_factor=1.3)
                    save_conversation(config['save_conversation'], user_input, best_answer)
            else:
                waring()
                break
        t=0
        ui.pushButton_2.setStyleSheet("background-color: rgb(255, 255, 255);")
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = MYUI()
    MainWindow.show()
    sys.exit(app.exec_())

