# Image Segmentation an STM32H7 Using UNet
Image segmentation on an STM32H7 STM32H7 microcontroller by implementing a UNet-inspired network. The network is designed and trained in Tensorflow and then deployed in C using STM32's X-Cube-AI library to import a .h5 file
  
The largest output of any layer is only 262 Kb and the board has 564 Kb of RAM. It runs inference at around 3-4 FPS by transfering images from PC to the MCU over USB.
  
[Project Video Demo](https://youtu.be/YPcGJSoCRz8?si=xzU44AGaH6TY4_KT)  
[Project Report](https://drive.google.com/file/d/1XYlqI2QfMuekKod-7ds-QvBYevuGWSJv/view?usp=sharing)  
[Dataset Link](https://github.com/anilsathyan7/Portrait-Segmentation), (search Portseg_128)  
[Dataset Link if the previous link does not work](https://drive.google.com/file/d/1UBLzvcqvt_fin9Y-48I_-lWQYfYpt_6J/view)  

## Predictions:

<img src="https://github.com/hershey890/Image_Segmentation_STM32/blob/main/readme_img1.png" width=300>
<img src="https://github.com/hershey890/Image_Segmentation_STM32/blob/main/readme_img2.png" width=300>
