## BaxterPlaysScrabble

Have you ever wanted to play scrabble with your Baxter robot? Well now you can!

By combining multiple scrabble solving AI with the robotic manipulation of the popular Baxter robot, BaxterPlaysScrabble is the first robot that will truly play scrabble with you; no digital boards or tablets needed.

This work represents the Masters of Engineering project by Sam Giampa and Ashish Bhatnagar at Cornell University.


# Playing the game & Testing

Begin the game by running ScrabbleGUI.py and follow on-screen instructions. Make no "opencv imshow" windows are spawning. 

boardvision.py and rackvision.py are used to configure the computer vision settings for the board and rack respectively. Calibration is neeeded for varying amounts of sunlight. 
We use a webcam pointed at both the board and rack to keep track of the state of the game. Board_vision.py and rack_vision.py interpret the feeds of the board and rack separately.
Please uncomment the "for testing" comments in board_vision.py and rack_vision.py to see view the current webcam view of the board and rack.

Classification.py loads a trained neural network. Once the model is loaded, Classification.classify is called for each tile on the Scrabble board and rack to classify its letter. 

Logic of maximizing points in each turn are located in board.py, solution.py, bag.py, direction.py, board_exceptions.py, and dictionary.py.

# Top-level Dependencies

OpenCV
Theano
Keras
Pillow
numpy
scipy
festival
baxter_interface

# Future Work

Improve computer vision accuracy. 
Improve deep learning accuracy. Training new, more specialized datasets may be helpful.

# Credit

The Scrabble basic maximizing algorithm was written by Lawrence Kesteloot. This library was altered to include prediction.

Github link: https://github.com/lkesteloot/scrabble