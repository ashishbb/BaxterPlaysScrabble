#!/usr/bin/python

import os
import Tkinter as tk
import tkFont
import tkSimpleDialog
from PIL import Image, ImageTk
import newmain
import direction


class UserInterface:

    def __init__(self):
        self.confirmedboard = [None]*15*15
        self.confirmedrack = [' ']*7
        self.root = tk.Tk()
        self.isred = True
        self.root.wm_title("Baxter Plays Scrabble")
        self.mrun = newmain.main_runner(self)
        self.firstboard = True
        self.currentturn = 0
        #constants and fonts
        self.helv36 = tkFont.Font(family='Helvetica',size=36, weight='bold') 
        self.helv24 = tkFont.Font(family='Helvetica',size=24, weight='bold')
        self.tileoffsetx = 67
        self.tileoffsety = 67
        self.rowcount = 0
        self.columncount = 0
        #board image
        self.image = Image.open("board.png")
        self.photo = ImageTk.PhotoImage(self.image)
        self.label = tk.Label(image=self.photo)
        self.label.image = self.photo
        self.label.grid(row=0,rowspan=2, column=0,columnspan=1,sticky=tk.E+tk.S+tk.W+tk.N)
        #text log box
        self.log= log_window(self.root)
        #nextmove button
        b = tk.Button(self.root, text="CLICK THIS WHEN READY TO START \n(WAIT AFTER PRESSING FOR BAXTER TO WARM UP)", 
            command=lambda:self.start(), bd=5,bg='#F00',fg='#FFF',font=self.helv24)
        b.grid(row=1, column=2,sticky=tk.E+tk.S+tk.W+tk.N)
        #labels
        self.label = tk.Label(self.root, text="YOUR RACK")
        self.label.place(x=1220,y=410)
        #mainloop
        self.root.mainloop()

    #add a lette to the board
    def addletter(self,letter,c,r,suggestion = False):
        if suggestion:
            self.letter = tk.Button(self.root, text=letter.upper(), 
                bg='#D3D3D3',fg='#8A360F',font=self.helv24)  
        else: 
            self.letter = tk.Button(self.root, text=letter.upper(), command=lambda:self.correctletter(c,r), 
                bg='#EECFA1',fg='#8A360F',font=self.helv24)
        self.letter.place(x=5+self.tileoffsetx*c,y=5+self.tileoffsety*r)

    #add a word to the board
    def addword(self,word,c,r,isdown,suggestion=False):
        count = 0
        print "isdown"
        print isdown
        if 'V' in str(isdown):
            for letter in word:
                self.addletter(word[count],c,r+count,suggestion)
                count += 1
        else:
            for letter in word:
                self.addletter(word[count],c+count,r,suggestion)
                count += 1       

        #callback function for correcting a letter on the board
    def correctletter(self,c,r):
        string = tkSimpleDialog.askstring('Letter Change', 'Please enter the new letter')
        if ' ' in string:
            self.confirmedboard[15*r+c] = None
        else:
            self.confirmedboard[15*r+c] = string[0].upper()
        self.addletter(string[0].upper(),c,r)

        #callback for performing the next set of main actions then switching gui button
    def swapbuttons(self):

        #Case 1: red button is pressed to indicate that physical board is ready for cv
        if self.isred:
            self.mrun.main2()
            b = tk.Button(self.root, text="CLICK THIS WHEN YOU HAVE\nFIXED ALL THE ERRONEOUS TILES\nAND THE BOARD IS CORRECT", 
                command=lambda:self.callback(), bd=5,bg='#66CD00',fg='#FFF',font=self.helv24)
            b.grid(row=1, column=2,sticky=tk.E+tk.S+tk.W+tk.N)
            os.system('cd /home/cs4752/ros_ws')
            os.system('rosrun chipmunks_proj3 chipmunkplayer.py')

        #Case 2: green button pressed to indicate board is correct on gui
        else:
            self.mrun.fixboard(self.confirmedboard)
            self.mrun.main3()
            self.currentturn += 1
            if self.currentturn >= 8:
                self.log.write('THE GAME IS OVER, THANK YOU')
                while True:
                    return
            self.mrun.main1()
            b = tk.Button(self.root, text="CLICK THIS WHEN YOU'VE FINISHED\nYOUR MOVE AND FILLED YOUR RACK\nWITH NEW TILES", 
                command=lambda:self.callback(), bd=5,bg='#F00',fg='#FFF',font=self.helv24)
            self.log.write('Next Turn')
            b.grid(row=1, column=2,sticky=tk.E+tk.S+tk.W+tk.N) 
        self.isred = not self.isred

        #sync gui board from vision only in spots that havent been confirmed by user
    def full_board_update(self,boardcells):
        for i in range(len(boardcells)):
            if self.confirmedboard[i] == None and boardcells[i] != None:
                self.confirmedboard[i] = boardcells[i]

        for i in range(len(self.confirmedboard)):
            c = i%15
            r = i/15
            if self.confirmedboard[i]==None:
                self.addletter(' ',c,r)
            else:
                self.addletter(self.confirmedboard[i],c,r)

        #display rack
    def show_rack(self,rack):
        for i in range(len(rack)):
            rackletter = tk.Button(self.root, text=rack[i].upper(), 
                command=lambda:self.correct_rack(i), bd=3,bg='#B11',fg='#FFF',font=self.helv24)
            rackletter.place(x=1060+self.tileoffsetx*i,y=470)

        ##TODO##
    def correct_rack(self,index):
        pass

    def callback(self):
        self.swapbuttons()

        #intial button callback
    def start(self):
        self.log.write('baxter is warming up his predictive skills, please wait a second')
        self.mrun.main0()
        os.system('cd /home/cs4752/ros_ws')
        os.system('rosrun chipmunks_proj3 chipmunkplayer.py')
        self.mrun.main1()
        b = tk.Button(self.root, text="CLICK THIS WHEN YOU'VE FINISHED\nYOUR MOVE AND FILLED YOUR RACK\nWITH NEW TILES", 
                command=lambda:self.callback(), bd=5,bg='#F00',fg='#FFF',font=self.helv24)
        b.grid(row=1, column=2,sticky=tk.E+tk.S+tk.W+tk.N) 

class log_window:
    def __init__(self,master):
        self.count = 0
        self.textframe = tk.Frame(master)
        self.text = tk.Text(self.textframe)
        self.text.insert(tk.END,'Welcome to BaxterPlaysScrabble')
        self.scrollbar = tk.Scrollbar(self.textframe,command=self.text.yview)
        self.text['yscrollcommand'] = self.scrollbar.set
        self.text.grid(row=0, column=2,sticky=tk.E+tk.N+tk.S+tk.W)
        self.textframe.grid(row=0, column=2,sticky=tk.E+tk.N+tk.S+tk.W)
        self.scrollbar.grid(row=0,column=2,sticky=tk.E+tk.N+tk.S)
    def write(self,text):
        self.count += 1
        self.text.insert(tk.END,'\n' + str(self.count) + ':' + text)
        self.text.see(tk.END)

us = UserInterface()
