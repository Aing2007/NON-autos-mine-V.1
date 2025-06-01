import sounddevice as sd
import numpy as np
import tensorflow as tf
import time
import requests
import serial  # เพิ่มการเชื่อมต่อ Serial
from scipy.io import wavfile

# กำหนดพารามิเตอร์
fs = 29354.667  # อัตราการสุ่มตัวอย่าง (sample rate)
duration = 1.5  # ความยาวของการบันทึก (วินาที)

# โหลดโมเดล TensorFlow Lite
interpreter = tf.lite.Interpreter(model_path="soundclassifier_with_metadata.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ตั้งค่า Serial Port
serial_port = 'COM9'  # เปลี่ยนเป็นพอร์ตที่ใช้จริง เช่น 'COM3' บน Windows หรือ '/dev/ttyUSB0' บน Linux
baud_rate = 115200
ser = serial.Serial(serial_port, baud_rate, timeout=1)

# ฟังก์ชันเล่นไฟล์เสียง
def play_audio(file_path):
    fs, audio_data = wavfile.read(file_path)
    if audio_data.ndim == 2:
        audio_data = audio_data[:, 0]  # เลือกช่องสัญญาณแรกหากมีหลายช่อง
    audio_data = np.array(audio_data, dtype=np.float32)
    audio_data = audio_data / np.max(np.abs(audio_data))  # Normalize ข้อมูลเสียง
    sd.play(audio_data, samplerate=fs)
    sd.wait()  # รอให้เล่นเสร็จ

# ฟังก์ชันส่งข้อความผ่าน LINE Notify
def send_line_notify(message, token):
    api_url = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {token}'}
    payload = {'message': message}
    response = requests.post(api_url, headers=headers, data=payload)
    return response.status_code

# ฟังก์ชันสำหรับการทำนายเสียง
def predict_sound(token):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # รอให้การบันทึกเสร็จ
    print("Recording complete.")

    input_shape = input_details[0]['shape']
    print("Expected input shape:", input_shape)

    # เตรียมข้อมูลอินพุต
    input_data = np.array(recording, dtype=np.float32)
    if len(input_shape) == 3:
        input_data = np.reshape(input_data, (1, -1, 1))
    elif len(input_shape) == 2:
        input_data = np.reshape(input_data, (1, -1))
    else:
        raise ValueError("Unsupported input shape")

    # ทำนายด้วยโมเดล
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    probabilities = output_data[0]
    class_index = np.argmax(probabilities)
    confidence = probabilities[class_index]
    
    # สมมุติว่ามีลิสต์ของชื่อคลาส
    class_names = [
        "Bully", "สวัสดี", "อายุเท่าไหร่", "สบายดีมั้ย",
        "เธอชื่ออะไร", "เสียงรบกวนเบื่องหลัง"]

    # แสดงผลลัพธ์
    print("ผลการทำนายที่มั่นใจที่สุด :", class_names[class_index])
    print("ความมั่นใจ :", confidence)
    int (confidence)
    
    # ถ้าผลลัพธ์คือ "Bully" ให้ส่งข้อความผ่าน LINE และเล่นไฟล์เสียง
    if class_names[class_index] == "Bully":
        if confidence >= 0.7:
         print("ส่งข้อความผ่าน LINE และเล่นไฟล์เสียง")
         message = 'ขณะนี้ลูกของคุณกำลังถูกกลั่นแกล้ง!!!'
         status_code = send_line_notify(message, token)
         print(f'Status Code: {status_code}')
         play_audio("Bully.wav")

    # เพิ่มการเล่นไฟล์เสียงสำหรับแต่ละคลาส
    elif class_names[class_index] == "สวัสดี":
        if confidence >= 0.7:
         play_audio("hi.wav")
         print("เล่นเสียงการตอบกลับ สำหรับการสวัสดี")

    elif class_names[class_index] == "เธอชื่ออะไร": 
        
         play_audio("name.wav")
         print("เล่นเสียงการตอบกลับ สำหรับเธอชื่ออะไร")

    elif class_names[class_index] == "หิวมั้ย":
        if confidence >= 0.7:
         play_audio("hungry.wav")
         print("เล่นเสียงการตอบกลับ สำหรับหิวมั้ย")

    elif class_names[class_index] == "เป็นอะไรหรือเปล่า":
        if confidence >= 0.7:
         play_audio("ok.wav")
         print("เล่นเสียงการตอบกลับ สำหรับเป็นอะไรรึเปล่า")

    elif class_names[class_index] == "สบายดีมั้ย":
        if confidence >= 0.7:
         play_audio("are you ok.wav")
         print("เล่นเสียงการตอบกลับ สำหรับสบายดีมั้ย")

    elif class_names[class_index] == "วันนี้เป็นยังไงบ้าง":
        if confidence >= 0.7:
         play_audio("how going on today.wav")
         print("เล่นเสียงการตอบกลับ สำหรับวันนี้เป็นยังไงบ้าง")

    elif class_names[class_index] == "อยากไปที่ไหน":
        if confidence >= 0.7:
         play_audio("where.wav")
         print("เล่นเสียงการตอบกลับ สำหรับอยากไปที่ไหน")
 
    elif class_names[class_index] == "วันนี้อากาศเป็นยังไง":
        if confidence >= 0.7:
         play_audio("weather.wav")
         print("เล่นเสียงการตอบกลับ สำหรับวันนี้อากาศเป็นยังไงบ้าง")

    elif class_names[class_index] == "อายุเท่าไหร่":
        if confidence >= 0.7:
         play_audio("old.wav")
         print("เล่นเสียงการตอบกลับ สำหรับอายุเท่าไหร่")
        

# ฟังก์ชันหลัก
# ฟังก์ชันหลัก
def main():
    token = 'WqiGBdulUANUQM3sskPGmklY1oMQB2BWUF8FrJ388m3'  # ใส่ token ของคุณที่นี่
    global checked_for_song  # ระบุว่าใช้ตัวแปร global
    checked_for_song = False  # กำหนดค่าเริ่มต้นเป็น False

    while True:
        if ser.in_waiting > 0:
            try:
                # อ่านข้อมูลเป็นไบต์
                serial_data_bytes = ser.readline()
                
                # พยายามถอดรหัสเป็น UTF-8
                try:
                    serial_data = serial_data_bytes.decode('utf-8').strip()
                except UnicodeDecodeError:
                    # ถ้าไม่สามารถถอดรหัสเป็น UTF-8 ให้ลองถอดรหัสเป็น 'latin-1'
                    try:
                        serial_data = serial_data_bytes.decode('latin-1').strip()
                    except UnicodeDecodeError:
                        print("Error decoding serial data: Data may not be 'latin-1' encoded")
                        continue

                print(f"Data from serial: {serial_data}")

                
                if serial_data == "เพลงกล่อมเด็ก":
                   value = int(serial_data)
                   if value >= 120:
                    play_audio("เสียงกล่อม.wav")  # เสียงเพลงกล่อมเด็ก
                    print("เล่นเพลง")
                
                if serial_data == "เพลงช้าง":
                   value = int(serial_data)
                   if value >= 120:
                    play_audio("เพลงช้าง.wav")  # เสียงเพลงกล่อมเด็ก
                    print("เล่นเพลง")


                # ตรวจสอบข้อความ
                if not checked_for_song:
                    if serial_data in ["เพลงกล่อมเด็ก", "เพลงช้าง"]:
                        print(f"พบข้อความ '{serial_data}'")
                        checked_for_song = True
                        # ข้ามรอบปัจจุบันของลูปเพื่อรอรับข้อมูลตัวเลขในรอบถัดไป
                        continue

                if checked_for_song:
                    try:
                        value = int(serial_data)
                        if value >= 110:
                            print("ชีพจรเต้นเร็ว!!")
                            # เล่นเสียงตามประเภทของเพลง
                            if serial_data == "เพลงกล่อมเด็ก":
                                play_audio("เสียงล่อม.wav")  # เสียงเพลงกล่อมเด็ก
                                print("เล่นเพลง")
                            elif serial_data == "เพลงช้าง":
                                play_audio("เพลงช้าง.wav")  # เสียงเพลงช้าง
                                print("เล่นเพลง")
                            elif serial_data == "เพลงสากล":
                                #play_audio("path/to/international_song.wav")  # เสียงเพลงสากล
                                print("เล่นเพลง")
                            elif serial_data == "เพลงดนตรีไทย":
                                #play_audio("path/to/thai_music.wav")  # เสียงเพลงดนตรีไทย
                                print("เล่นเพลง")
                                
                            print(f"ค่า {value} >= 120, กำลังเล่นเสียงสำหรับ '{serial_data}'")
                            # รีเซ็ตสถานะหลังจากเล่นเสียง
                            checked_for_song = False
                    except ValueError:
                        print("ข้อมูลที่ได้รับไม่ใช่ตัวเลข")

            except Exception as e:
                print(f"Error processing serial data: {e}")
        predict_sound(token)        
        time.sleep(1)  # รอ 1 วินาทีก่อนทำการบันทึกและทำนายอีกครั้ง

if __name__ == "__main__":
    main()
