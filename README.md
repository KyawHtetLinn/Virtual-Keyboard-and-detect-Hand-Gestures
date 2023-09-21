# Virtual Keyboard and detecting hand gestures with Opencv

  In this repo, we are going to see a hand gestures detection and an AI virtual keyboard on screen.The model can predict left or right hand and close or open.
The model also track the index finger for the keywords on the screen.

https://github.com/KyawHtetLinn/Virtual-Keyboard-and-detect-Hand-Gestures/assets/70162137/bd2c693c-1e97-4ada-9ffd-7b1cda95ecfa

If you close index finger and middel finger while placing index finger on a keyword,it will regard as clicking.The history of the pointer will show green circles.
For detecting the finger points , I tried the cvzone model ,but may be due to my web cam or resolution the model didn't work and show the fingers points correctly. So, I used the model from [this repo](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe) .The results are better and it estimates correctly.



![alt text](https://user-images.githubusercontent.com/37477845/102242918-ed328c80-3f3d-11eb-907c-61ba05678d54.png)

