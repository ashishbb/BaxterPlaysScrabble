# BaxterPlaysScrabble

Have you ever wanted to play scrabble with your Baxter robot? Well now you can!

By combining multiple scrabble solving AI with the robotic manipulation of the popular Baxter robot, BaxterPlaysScrabble is the first robot that will truly play scrabble with you; no digital boards or tablets needed.

WORK IN PROGRESS

This work represents the Masters of Engineering project by Sam Giampa and Ashish Bhatnagar at Cornell University.


Playing the game

Begin the game by running ScrabbleGUI.py and follow on-screen instructions. Make no "opencv imshow" windows are spawning. 

boardvision.py and rackvision.py are used to configure the computer vision settings for the board and rack respectively. Calibration is neeeded for varying amounts of sunlight. 

Classification.py loads a trained neural network. Once the model is loaded, Classification.classify is called for each tile on the Scrabble board and rack to classify its letter. 

Logic of maximizing points in each turn are located in board.py, solution.py, bag.py, direction.py, and dictionary.py.


Top-level Dependencies

OpenCV
Theano
Keras
Pillow
numpy
scipy
festival

