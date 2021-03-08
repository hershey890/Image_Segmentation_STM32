import numpy as np
import cv2
from time import time
import serial
import argparse


class Face_Seg():
    def __init__(self, mode):
        self.mode = mode.strip()

    def _init_serial(self):
        ser = serial.Serial(
            port='COM4',
            baudrate=115200,
            parity=serial.PARITY_ODD,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.SIXBITS
        )
        ser.isOpen()
        return ser

    def main_loop(self):
        x_len, y_len = 64, 64
        cap, ind_y1, ind_y2, background_img = None, None, None, None
        if self.mode == 'webcam':
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise(IOError("Cannot open webcam"))
            x_len_webcam, y_len_webcam = int(cap.get(3)), int(cap.get(4))
            ind_y1, ind_y2 = x_len_webcam//2 - y_len_webcam//2, x_len_webcam//2 + y_len_webcam//2
            background_img = cv2.imread('royce.png', cv2.IMREAD_COLOR)
        elif self.mode == 'dataset':
            img_dir = './Portseg_128/'
            images_rgb = np.load(img_dir + 'test_xtrain.npy')
            masks = np.load(img_dir + 'test_ytrain.npy')
            N, _, _, _ = images_rgb.shape
            images_gray = np.zeros((N, y_len, x_len), dtype=np.ubyte)
            for i in range(N):
                images_gray[i] = cv2.cvtColor(images_rgb[i], cv2.COLOR_RGB2GRAY)[::2,::2]

        ser = self._init_serial()

        RxTx_len = 64
        num_packets = x_len * y_len // RxTx_len
        Rx_buff = np.zeros(y_len*x_len, dtype=np.ubyte)
        num_packets_sent = 0
        sent_msg = False

        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 600,600)

        time0 = time()
        counter = 0
        Tx_buff = None
        out = None
        while 1 :
            if not sent_msg:
                if self.mode == 'webcam':
                    ret, frame = cap.read()
                    frame_crop = frame[:,ind_y1:ind_y2,:]
                    out = cv2.resize(frame_crop, (128,128))
                    Tx_buff = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)[::2,::2].flatten().tostring()
                elif self.mode == 'dataset':
                    Tx_buff = images_gray[counter].flatten().tostring()
                time0 = time()
                for i in range(num_packets):
                    ser.write(Tx_buff[i*RxTx_len:(i+1)*RxTx_len])
                sent_msg = True

            if ser.inWaiting() > 0:
                temp = list(ser.read(RxTx_len))
                Rx_buff[num_packets_sent*RxTx_len:(num_packets_sent+1)*RxTx_len] = np.array(temp, dtype=np.ubyte)
                num_packets_sent += 1

                if num_packets_sent == num_packets:
                    parsed_image = Rx_buff.reshape((x_len,y_len))
                    parsed_image = cv2.resize(parsed_image, (x_len*2, y_len*2))
                    if self.mode == 'webcam':
                        mask = np.dstack([parsed_image, parsed_image, parsed_image])
                        display_img = background_img * (1 - mask) + mask*out
                    elif self.mode == 'dataset':
                        parsed_image *= 255
                        mask = np.dstack([parsed_image*0, parsed_image*0, parsed_image])
                        display_img = np.flip(images_rgb[counter], 2) # OpenCV imshow takes BGR not RGB
                        display_img = cv2.addWeighted(display_img, 0.7, mask, 0.3, 0)
                    cv2.imshow('image', display_img)
                    cv2.waitKey(1)
                    counter += 1
                    sent_msg = False
                    num_packets_sent = 0
                    print("Time Spent:", time()-time0)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str)
    args = parser.parse_args()
    
    face_seg = Face_Seg(mode=args.mode)
    face_seg.main_loop()