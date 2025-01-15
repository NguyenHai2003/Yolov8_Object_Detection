from flask import Flask, render_template, Response, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
from ultralytics import YOLO
import cvzone
import math
from sort import *
from utils import *
from flask import flash, redirect, url_for

# Khởi tạo Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Cấu hình thư mục tạm để lưu video và ảnh tải lên
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Tải mô hình YOLOv8 đã huấn luyện sẵn
model = YOLO("../Yolo-Weights/yolov8n.pt")

# Danh sách các lớp vật thể trong COCO dataset


# Biến toàn cục để lưu video đầu vào
cap = None
image = None

mask = cv2.imread("data/nonFilter.png")


tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limits = [400, 297, 673, 297]
totalCount = []
delay_time = 0  # Tốc độ mặc định
current_input = 0 # 1: video, 2: webcam, 3: image
found_object =[]



def reset_globals():
    global cap, image, totalCount, current_input,found_object
    if cap is not None:
        cap.release()  # Giải phóng camera hoặc video
    cv2.destroyAllWindows()  # Đóng tất cả cửa sổ OpenCV
    cap = None
    image = None
    totalCount = []
    current_input = 0
    found_object = []
    print("All processes have been reset.")

def generate_frames_for_videos():
    global cap, current_input, totalCount
    while cap:
        success, img = cap.read()
        mask_resized = cv2.resize(mask, (int(img.shape[1]), int(img.shape[0])))
        imgRegion = cv2.bitwise_and(img, mask_resized)

        '''imgGraphics = cv2.imread("data/graphics.png", cv2.IMREAD_UNCHANGED)
        img = cvzone.overlayPNG(img, imgGraphics, (0, 0))'''
        results = model(imgRegion, stream=True)

        detections = np.empty((0, 5))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
                w, h = x2 - x1, y2 - y1

                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                if found_object:
                    for temp in found_object:
                        if temp == "all":
                            currentArray = np.array([x1, y1, x2, y2, conf])
                            detections = np.vstack((detections, currentArray))
                            break
                        if currentClass == temp and conf > 0.3:
                            # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                            #                    scale=0.6, thickness=1, offset=3)
                            # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                            currentArray = np.array([x1, y1, x2, y2, conf])
                            detections = np.vstack((detections, currentArray))
                else:
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))

        resultsTracker = tracker.update(detections)

        #cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(result)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
            cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                               scale=2, thickness=3, offset=10)

            if totalCount.count(id) == 0:
                totalCount.append(id)
        cv2.putText(img, "Total: " +str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

        # Chuyển đổi hình ảnh sang định dạng mà Flask có thể hiển thị
        ret, buffer = cv2.imencode('.jpg', img)
        if not ret:
            break
        frame = buffer.tobytes()

        time.sleep(delay_time)

        # Trả về hình ảnh dưới dạng byte
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')



def generate_frames_for_images():
    global totalCount
    while True:
        # Tải ảnh gốc từ nguồn
        original_image = image.copy()  # Tạo bản sao của ảnh gốc để giữ nguyên trạng thái ban đầu
        totalCount = []
        results = model(original_image)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)


                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                if found_object:
                    for temp in found_object:
                        if currentClass == temp:
                            totalCount.append(0)
                            cvzone.cornerRect(original_image, (x1, y1, x2 - x1, y2 - y1))
                            cvzone.putTextRect(original_image, f'{classNames[cls]} {conf}',
                                               (max(0, x1), max(35, y1)), scale=5, thickness=8)
                else:
                    totalCount.append(0)
                    cvzone.cornerRect(original_image, (x1, y1, x2 - x1, y2 - y1))
                    cvzone.putTextRect(original_image, f'{classNames[cls]} {conf}',
                                       (max(0, x1), max(35, y1)), scale=5, thickness=8)
        cv2.putText(original_image, "Total: " + str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

        # Chuyển ảnh đã xử lý thành định dạng có thể hiển thị trên web
        ret, buffer = cv2.imencode('.jpg', original_image)  # Lưu ảnh xử lý hiện tại
        if not ret:
            return 'Error processing image', 500
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


# Định tuyến chính để hiển thị video
@app.route('/video')
def video():
    if cap is not None:
        if(current_input == 1 or current_input == 2): #video or webcam
            return Response(generate_frames_for_videos(), mimetype='multipart/x-mixed-replace; boundary=frame')
        '''if (current_input == 2):  # video or webcam
            return Response(generate_frames_for_webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')'''
    elif image is not None: #image
        return Response(generate_frames_for_images(), mimetype='multipart/x-mixed-replace; boundary=frame')
    return 'No file uploaded', 400

# Định tuyến trang chủ
@app.route('/')
def index():
    return render_template('index.html')

# Định tuyến để nhận video hoặc ảnh từ người dùng
@app.route('/upload', methods=['POST'])
def upload():
    global cap, image, current_input
    reset_globals()
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        if file.filename.lower().endswith(('mp4', 'avi', 'mov')):
            cap = cv2.VideoCapture(file_path)
            image = None
            current_input = 1
        elif file.filename.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif')):
            image = cv2.imread(file_path)
            cap = None
            current_input = 3
        return render_template('index.html')


@app.route('/upload_filter', methods=['POST'])
def upload_filter():
    global mask
    if 'filter' not in request.files:
        flash('No filter file part', 'error')  # Thông báo lỗi
        return redirect(url_for('index'))

    filter_file = request.files['filter']
    if filter_file.filename == '':
        flash('No selected filter file', 'error')  # Thông báo lỗi
        return redirect(url_for('index'))

    if filter_file:
        filename = secure_filename(filter_file.filename)
        filter_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        filter_file.save(filter_path)

        # Cập nhật mask với file mới
        mask = cv2.imread(filter_path)
        if mask is None:
            flash('Invalid mask file', 'error')  # Thông báo lỗi
            return redirect(url_for('index'))

        flash('Filter uploaded and applied successfully', 'success')  # Thông báo thành công
        return redirect(url_for('index'))


@app.route('/webcam')
def webcam():
    global cap, current_input
    reset_globals()

    '''phone_camera_url = "http://192.168.1.179:4747/video"
    cap = cv2.VideoCapture(phone_camera_url)'''

    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("Failed to open the webcam.")
        return "Could not access the phone camera. Check the URL or connection.", 400

    print("Opened webcam")
    current_input = 2
    return render_template('index.html')


@app.route('/cancel', methods=['POST'])
def cancel_process():
    reset_globals()
    flash("Process has been cancelled successfully.", "success")
    return redirect("/")


@app.route('/handle_message', methods=['POST'])
def handle_message():
    global found_object,totalCount
    message = request.json['message']
    objects =""
    found_object = extract_entities(message)
    if found_object:
        print("Found objects:", found_object)
        totalCount = []
        objects = ", ".join(found_object.keys())
        message = message + "<object>"
    intents_list = predict_class(message)
    response = get_response(intents_list, objects)

    return jsonify({'response': response})


# Chạy ứng dụng Flask
if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)