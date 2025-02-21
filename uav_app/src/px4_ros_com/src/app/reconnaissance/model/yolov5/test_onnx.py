# import torch
# import cv2
# import numpy as np

# # ./runs/train/exp2/weights/best.pt

# # Step 1: Load the YOLOv5 model (custom or pretrained)
# model = torch.hub.load('', 'custom', path='yolov5n.pt', source='local')  # Make sure best.pt is in the correct path

# # Step 2: Load an image
# image_path = "cv_frame.jpg"  # Update the image path accordingly
# image = cv2.imread(image_path)

# # Step 3: Run inference (detect objects in the image)
# results = model(image)

# # Step 4: Extract bounding boxes, class labels, and confidence scores
# # results.xyxy[0] contains the bounding boxes and class info
# # Columns: xmin, ymin, xmax, ymax, confidence, class, name
# boxes = results.xyxy[0].cpu().numpy()

# # Step 5: Draw bounding boxes and labels on the image
# for box in boxes:
#     x1, y1, x2, y2, conf, class_id = map(float, box[:6])
#     label = results.names[class_id]
#     text = f'{label} {conf:.2f}'  # Class label and confidence

#     # Draw rectangle and label on the image
#     cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green color for bounding box
#     cv2.putText(image, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# # Step 6: Display the image with the drawn boxes
# cv2.imshow("Detection Results", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import cv2
import numpy as np

net = cv2.dnn.readNetFromONNX("./runs/train/exp2/weights/best.onnx")

img = cv2.imread("cv_frame.jpg")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = img.astype(np.float32) / 255.0

# Preprocess the image and create blob
blob = cv2.dnn.blobFromImage(img, 1/255.0, (640, 640), (0, 0, 0), swapRB=True, crop=False)

# Set input for the network
net.setInput(blob)

# Perform inference
outputs = net.forward()

detections = outputs.squeeze()

# Confidence threshold to filter out low confidence detections
confidence_threshold = 0.5

# NMS threshold
nms_threshold = 0.4

# Extract detections and filter by confidence
filtered_detections = []

for detection in detections:
    class_id, confidence, x_center, y_center, width, height = detection[:6]
    if confidence > confidence_threshold:
        filtered_detections.append(detection)

# Convert the filtered detections to bounding boxes
boxes = []
for detection in filtered_detections:
    class_id, confidence, x_center, y_center, width, height = detection[:6]
    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)
    boxes.append([x_min, y_min, x_max, y_max, confidence, class_id])

# Apply Non-Maximum Suppression (NMS)
def nms(boxes, nms_threshold):
    if len(boxes) == 0:
        return []

    # Sort boxes by confidence
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)

    nms_boxes = []
    while boxes:
        best_box = boxes.pop(0)
        nms_boxes.append(best_box)
        boxes = [
            box
            for box in boxes
            if iou(best_box, box) < nms_threshold
        ]

    return nms_boxes

def iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max, _, _ = box1
    x2_min, y2_min, x2_max, y2_max, _, _ = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min + 1) * max(0, inter_y_max - inter_y_min + 1)

    box1_area = (x1_max - x1_min + 1) * (y1_max - y1_min + 1)
    box2_area = (x2_max - x2_min + 1) * (y2_max - y2_min + 1)

    iou = inter_area / float(box1_area + box2_area - inter_area)

    return iou

nms_boxes = nms(boxes, nms_threshold)

# Now nms_boxes contains the final filtered and post-processed bounding boxes
for box in nms_boxes:
    x_min, y_min, x_max, y_max, confidence, class_id = box
    print(f'Class ID: {class_id}, Confidence: {confidence}, Bounding Box: [{x_min}, {y_min}, {x_max}, {y_max}]')
