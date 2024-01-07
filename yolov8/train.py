from ultralytics import YOLO

data_path=''
model_path=''
number_of_epoch=
# Load a model
model = YOLO("yolov8x.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data=data_path, epochs=number_of_epoch)

model.val(data=data_path)