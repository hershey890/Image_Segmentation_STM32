import numpy as np
import cv2 as cv
from time import time
import serial
import sys
import matplotlib.pyplot as plt

# Load the data you want to send
img_dir = './Portseg_128/'
images_rgb = np.load(img_dir + 'test_xtrain.npy')
masks = np.load(img_dir + 'test_ytrain.npy')
N, y_len, x_len, _ = images_rgb.shape
x_len = x_len // 2
y_len = y_len // 2
images_gray = np.zeros((N, y_len, x_len), dtype=np.ubyte)
for i in range(N):
  images_gray[i] = cv.cvtColor(images_rgb[i], cv.COLOR_RGB2GRAY)[::2,::2]

# The H7 mirrors whatever you put here so you only need to change the com port
ser = serial.Serial(
    port='COM4',
    baudrate=115200,
    parity=serial.PARITY_ODD,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.SIXBITS
)

ser.isOpen()

RxTx_len = 64
num_packets = x_len * y_len // RxTx_len
Rx_buff = np.zeros(y_len*x_len, dtype=np.ubyte)
target_freq = 1 / 10
num_packets_sent = 0
num_bytes_received = 0
sent_msg = False

cv.namedWindow('image', cv.WINDOW_NORMAL)
cv.resizeWindow('image', 600,600)

time0 = time()
counter = 0
while 1 :
    time_diff = time() - time0
    if time_diff >= target_freq and not sent_msg:
        Tx_buff = images_gray[counter].flatten().tostring()
        time0 = time()
        for i in range(num_packets):
            ser.write(Tx_buff[i*RxTx_len:(i+1)*RxTx_len])
        sent_msg = True

    if ser.inWaiting() > 0:
        temp = list(ser.read(RxTx_len))
        Rx_buff[num_packets_sent*RxTx_len:(num_packets_sent+1)*RxTx_len] = np.array(temp, dtype=np.ubyte)
        num_packets_sent += 1

        # Parsing on H7 Done and Data in Rx_buff
        if num_packets_sent == num_packets:
            parsed_image = Rx_buff.reshape((x_len,y_len)) * 255
            parsed_image = cv.resize(parsed_image, (x_len*2, y_len*2))
            mask = np.dstack([parsed_image*0, parsed_image*0, parsed_image])
            display_img = np.flip(images_rgb[counter], 2) # OpenCV imshow takes BGR not RGB
            # Displaying image
            cv.imshow('image', cv.addWeighted(display_img, 0.7, mask, 0.3, 0))
            cv.waitKey(1)
            plt.imshow(mask)
            counter += 1
            sent_msg = False
            num_packets_sent = 0