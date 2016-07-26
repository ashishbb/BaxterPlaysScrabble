#!/usr/bin/env python

import sys
import rospy
import numpy as np
import math
#import baxter_interface
import time

from board import Board
from dictionary import Dictionary
from bag import generate_rack, get_full_bag

import cv2
#import speech
import copy
import dictionary
import direction
import board
import solution
import board_vision
import rack_vision
from Classification import CNN_Model
import festival

DICTIONARY_FILENAME = "dictionary"

class main_runner:

    def __init__(self,gui):
        self.gui = gui

    def main0(self):

        #gui = ScrabbleGUI.UserInterface()

        # Load the dictionary.
        self.dictionary = Dictionary.load(DICTIONARY_FILENAME)
        self.board = Board()

        # Keep track of the winning solution at each round.
        self.winners = []

        # List of letters we can still pick from.
        self.bag = get_full_bag()

        # Rack starts out empty. Keep track of current and last rack state.
        self.rack = ""
        self.old_rack = ""
        self.count = 0

        # Keep track of current and last board state,
        self.update_board = None
        self.old_board = Board()

        # Baxter's score
        self.my_score = 0

        #Create classifier
        self.classify = CNN_Model()

        # set Baxter's mode.
        # mode = 0: skill level tapers off and stays low
        # mode = 1: skill level tapers off and increases after reaching lowest point
        # mode = 2: highest skill level for whole game 
        self.mode = 0

        festival.setStretchFactor(1)
        festival.sayText("Hello, it is Baxter here, I hope we do well")

        # Create video feeds
        # cam = cv2.VideoCapture(1)
        # print cam.isOpened()

    def main1(self):

        # Keep playing until we're out of tiles or solutions.
        self.count+=1
        # Fill up our rack.
        print "Bag: %s" % "".join(self.bag)
        self.old_rack = self.rack

        # Updates rack with current rack from video feed.
        # cam1 = cv2.VideoCapture(1)
        # print cam1.isOpened()
        self.cam1 = 1
        self.rack1 = rack_vision.get_rack(self.classify,self.cam1)
        self.rack2 = rack_vision.get_rack(self.classify,self.cam1)
        iteration = 0
        while (not (self.rack1 in self.rack2)) and iteration <=6:
            self.rack1 = rack_vision.get_rack(self.classify,self.cam1)
            self.rack2 = rack_vision.get_rack(self.classify,self.cam1)
            iteration += 1

        self.rack = self.rack1
        self.gui.show_rack(self.rack)

        # cam1.release()
        cv2.destroyAllWindows()

        print("RACK:")
        print(self.rack)

        # Get a list of possible solutions. These aren't all necessarily legal.
        self.solutions = self.board.generate_solutions(self.rack, self.dictionary)


        # self.solution = self.board.find_best_solution(self.solutions, self.dictionary)

        self.solution = self.board.solution_curve(self.solutions,self.dictionary,self.mode,self.count)
        #print('SOLUTIONSSSS %s' %self.solutions)

        self.gui.addword(self.solution.word,self.solution.col,self.solution.row,self.solution.direction,suggestion = True)
        festival.sayText("I think you should play")
        festival.setStretchFactor(1.4)
        festival.sayText(self.solution.word)


    def main2(self):
        print 'old board at the beginningof 2'
        print self.old_board

        if self.solution:
            print "Winner: %s" % self.solution

            # Play the winning solution.
            self.board.create_board()
            print("I suggest you play the word:"+self.solution.word)

            #speech.speak(solution)
        else:
            pass
            # Should put letters back in bag.
            
        print board

        print 'old board at the beginningof 2 - 1'
        print self.old_board
        
        # Wait for "Enter", signifying the player has completed his/her turn.
        #self.wait = raw_input("Press enter when finished with move")

        # Get word that was just played on the board by fetching the new board state.
        self.update_board = Board()

        print 'old board at the beginningof 2 - 2'
        print self.old_board

        board1 = board_vision.get_board(self.classify,self.cam1)
        board2 = board_vision.get_board(self.classify,self.cam1)
        board3 = board_vision.get_board(self.classify,self.cam1)

        print 'old board at the beginningof 2 - 3'
        print self.old_board
        correctboard = [None] *15*15
        for i in range(len(board1)):
            if board1[i]==board2[i] or board1[i]==board3[i]:
                correctboard[i] = board1[i]
            elif board2[i]==board3[i]:
                correctboard[i] = board2[i]

        self.update_board.set_cells(correctboard)

        print 'old board at the beginningof 2 - 4'
        print self.old_board


        self.gui.full_board_update(copy.deepcopy(self.update_board.cells))

        print 'old board at the end gof 2'
        print self.old_board

    def main3(self): 
        print"GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG"
        print self.update_board
        print self.old_board
        print 'Gassadasdasdasdadasdasdasdasdasdasdasdasdasdasdasd'
        self.move,self.letter_placed_on_board = self.board.get_played_word(self.update_board,self.old_board,self.dictionary)

        print ("The word:"+ self.move.word +"was just played.")

        for char in self.letter_placed_on_board:
            self.rack = self.rack.replace(char,"")

        #rack_list = self.letter_placed_on_board.split()
        self.move.rack_indices = self.letter_placed_on_board
        self.board.add_solution(self.move)

        if (self.move.word == self.solution.word):
            print("Player listened to Baxter")
        else:
            print("Bitch defied Baxter")

        if (self.move.score != None):
            self.my_score+=self.move.score
            if len(self.move.rack_indices) == 7:
                self.my_score+=50

        print "Baxter's Score: %d" % self.my_score
        festival.setStretchFactor(1)
        festival.sayText("Good Move")
        self.gui.log.write('Good job, your score is now: ' + str(self.my_score))

        #generate_rack(self.rack,self.old_rack,self.bag)

        for char in self.letter_placed_on_board:
            self.rack = self.rack.replace(char,"")
        self.old_board = Board()
        self.old_board.set_cells(self.update_board.cells)
        print 'oldboard set to'
        print self.old_board
        print ("count:"+str(self.count))

    def main4(self):
        print "Baxter's Score: %d" % self.my_score
        print "Baxter's Words:"
        for self.rack,self.winner in self.winners:
            print "    %s: %s" % (self.rack, self.winner)

    def fixboard(self,boardcells):
        print 'old board in this bitch1'
        print self.old_board
        self.update_board = Board()
        self.update_board.set_cells(copy.deepcopy(boardcells))
        print 'old board in this bitch'
        print self.old_board


# if __name__ == "__main__":
#     try:
#         main2()
#     except rospy.ROSInterruptException:
#         pass