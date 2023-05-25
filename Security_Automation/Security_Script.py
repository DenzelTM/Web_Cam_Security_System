#By: Denzel Musiyiwa
import cv2
import smtplib
import datetime
import imghdr
import os
from email.message import EmailMessage


#Set up your email
sender_email = 'youremail@gmail.com'
sender_password = 'yourpassword'
receiver_email = 'email1@gmail.com'
smtp_server = 'smtp.gmail.com'
smptp_port = 587

#initialize webcam
video = cv2.VideoCapture(0)

#Initialize motion detector
previous_frame = None
motion_detected = False

while True:
    #Read the current frame from webcam
    ret, frame = video.read()

    #Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Apply Gaussian blur to reduce the noise
    gray= cv2.GaussianBlur(gray, (21, 21), 0)
#set previous frame for motion detector
    if previous_frame is None:
        previous_frame = gray
        continue
    
    #calculate difference between current and previous frame
    frame_delta = cv2.absdiff(previous_frame, gray)

    #Apply threasholding to highlight places with the biggest difference
    threash = cv2.threshold(frame_delta, 30, 255, cv2.THRESH_BINARY)[1]

    threash = cv2.dilate(threash, None, iterations=2)
    contours, _ = cv2.findContours(threash.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Check for motion detection
    motion_detected = False
    for contour in contours:
          if cv2.contourArea(contour) < 500:
               continue
          (x, y, w, h) = cv2.boundingRect(contour)
          cv2.rectangle(frame, (x,y), (x + w , y + h), (0, 255, 0), 2)
          motion_detected = True
    #Display the resulting frame
    cv2.imshow("Home security system", frame)

    #Capture and send image if motion is detected
    if motion_detected:
         timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
         image_path = f"motion_{timestamp}.jpg"
         cv2.imwrite(image_path, frame)

       #prepare email message
         message = EmailMessage()
         message['Subject'] = f"Motion Detected at {timestamp}"
         message['From'] = sender_email
         message['To'] = receiver_email

        #Attach Captured image to email
    with open(image_path, 'rb') as attachment:
       image_data = attachment.read()
       image_type = imghdr.what(attachment.name)
       message.add_attachment(image_data, maintype= 'image', ubtype=image_type, filename=attachment.name)

     #send the email
    with smtplib.SMTP(smtp_server, smptp_port) as server:
         server.starttls()
         server.login(sender_email, sender_password)
         server.send_message(message)

    #Delete the image file
    os.remove(image_path)
  
#release the webcam and close windows
video.release()
cv2.destroyAllWindows()







              



