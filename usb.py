import numpy as np
import cv2
from time import time, sleep
import serial
import argparse
from threading import Thread

class Face_Seg():
    def __init__(self, mode):
        self.mode = mode.strip()
        # Webcam runs in a seperate thread
        if self.mode == 'webcam':
            self.cap = cv2.VideoCapture(0)
            while not self.cap.isOpened: pass
            self.thread = Thread(target=self._update_frame, args=())
            self.thread.daemon = True
            self.thread.start()

    def _update_frame(self):
        # Only used in webcam mode
        # Reads the next frame from the webcam in a seperate thread
        while 1:
            (self.status, self.frame) = self.cap.read()
            sleep(.01) 

    def main_loop(self):
        # Initialization
        x_len, y_len = 64, 64

        # Import webcam background
        cap, ind_y1, ind_y2, background_img = None, None, None, None
        if self.mode == 'webcam':
            x_len_webcam, y_len_webcam = int(self.cap.get(3)), int(self.cap.get(4))
            ind_y1, ind_y2 = x_len_webcam//2 - y_len_webcam//2, x_len_webcam//2 + y_len_webcam//2
            background_img = cv2.imread('royce.png', cv2.IMREAD_COLOR)
            # background_img = cv2.imread('tiger_king_big.png', cv2.IMREAD_COLOR)
            if background_img is None: raise(IOError('Input Image Error'))

        # Import dataset
        elif self.mode == 'dataset':
            img_dir = './Portseg_128/'
            images_rgb = np.load(img_dir + 'test_xtrain.npy')
            masks = np.load(img_dir + 'test_ytrain.npy')
            N, _, _, _ = images_rgb.shape
            images_gray = np.zeros((N, y_len, x_len), dtype=np.ubyte)
            for i in range(N):
                images_gray[i] = cv2.cvtColor(images_rgb[i], cv2.COLOR_RGB2GRAY)[::2,::2]

        # Initiate COM Port/Serial
        ser = serial.Serial(
            port='COM4',
            baudrate=115200,
            parity=serial.PARITY_ODD,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.SIXBITS
        )
        ser.isOpen()

        # OpenCV Window Settings
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 600,600)
        if self.mode == 'webcam':
            if background_img.shape[0] != 128:
                cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('image', 640*2,480*2)

        RxTx_len = 64
        Tx_num_packets = x_len * y_len // RxTx_len
        Rx_num_packets = 8 # 64 * 64 / (RxTx_len * sizeof(uint_8))
        Rx_buff = np.zeros(y_len*x_len, dtype=np.ubyte)
        num_packets_received = 0
        sent_msg = False
        time0 = 0
        counter = 0
        Tx_buff = None
        out = None
        mask = np.ones_like(background_img, dtype=np.ubyte)
        old_frame = None
        new_mask_ready = False

        # Wait for the Webcam/Webcam Thread to be Ready
        frame_thread_ready = True
        if self.mode == 'webcam':
            while 1:
                frame_thread_ready = True
                try:
                    frame = self.frame
                except AttributeError:
                    frame_thread_ready = False
                if frame_thread_ready: break

        # Run network
        while 1 :

            # Read from Webcam
            if self.mode == 'webcam':
                frame = self.frame
                frame_crop = frame[:,ind_y1:ind_y2,:]
                out = cv2.resize(frame_crop, (128,128))
                if background_img.shape[0] == 128 and background_img.shape[1] == 128:
                    display_img = background_img - background_img * mask + mask * out
                else:
                    if new_mask_ready:
                        mask = cv2.resize(mask, (400, 400)) #640, 480
                        mask = cv2.copyMakeBorder(mask, 80, 0, 120, 120, borderType=cv2.BORDER_CONSTANT, value=0)
                        new_mask_ready = False                    
                    display_img = background_img - background_img * mask + mask * frame
                cv2.imshow('image', display_img)
                if cv2.waitKey(1) == ord('q'):
                    self.cap.release()
                    cv2.destroyAllWindows()
                    exit()

            # Transmitting
            if not sent_msg:
                if self.mode == 'webcam':
                    Tx_buff = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)[::2,::2].flatten().tostring()
                elif self.mode == 'dataset':
                    Tx_buff = images_gray[counter].flatten().tostring()
                time0 = time()
                for i in range(Tx_num_packets):
                    ser.write(Tx_buff[i*RxTx_len:(i+1)*RxTx_len])
                sent_msg = True

            # Receiving
            if ser.inWaiting() > 0:
                Rx_word = ser.read(RxTx_len)
                for i in range(64): #512=64*8
                    Rx_buff[num_packets_received*8*RxTx_len + 8*i + 0] = Rx_word[i] & 0b00000001
                    Rx_buff[num_packets_received*8*RxTx_len + 8*i + 1] = (Rx_word[i] & 0b00000010) >> 1
                    Rx_buff[num_packets_received*8*RxTx_len + 8*i + 2] = (Rx_word[i] & 0b00000100) >> 2
                    Rx_buff[num_packets_received*8*RxTx_len + 8*i + 3] = (Rx_word[i] & 0b00001000) >> 3
                    Rx_buff[num_packets_received*8*RxTx_len + 8*i + 4] = (Rx_word[i] & 0b00010000) >> 4
                    Rx_buff[num_packets_received*8*RxTx_len + 8*i + 5] = (Rx_word[i] & 0b00100000) >> 5
                    Rx_buff[num_packets_received*8*RxTx_len + 8*i + 6] = (Rx_word[i] & 0b01000000) >> 6
                    Rx_buff[num_packets_received*8*RxTx_len + 8*i + 7] = (Rx_word[i] & 0b10000000) >> 7

                num_packets_received += 1

                # Is the message is fully received from the MCU
                if num_packets_received == Rx_num_packets:
                    raw_map = Rx_buff.reshape((x_len,y_len))
                    raw_map = cv2.resize(raw_map, (x_len*2, y_len*2))

                    if self.mode == 'webcam':
                        mask = np.dstack([raw_map, raw_map, raw_map])
                        new_mask_ready = True

                    elif self.mode == 'dataset':
                        mask = np.dstack([raw_map*0, raw_map*0, raw_map*255])
                        display_img = np.flip(images_rgb[counter], 2)
                        display_img = cv2.addWeighted(display_img, 0.7, mask, 0.3, 0)
                        cv2.imshow('image', display_img)
                        if cv2.waitKey(1) == ord('q'):
                            cv2.destroyAllWindows()
                            exit()
                        counter += 1

                    sent_msg = False
                    num_packets_received = 0
                    print("Time Spent:", time()-time0)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str)
    args = parser.parse_args()
    
    face_seg = Face_Seg(mode=args.mode)
    face_seg.main_loop()