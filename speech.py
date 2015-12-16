import pyttsx

def speak(solution):
    direction = solution.direction
    if direction == "V":
        play_direction = "vertically"
    else:
        play_direction = "horizontally"

    engine = pyttsx.init()
    engine.setProperty('voice', "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\VW Kate")#if I don't do this line then it says both the commands
    engine.say("Play %s, %s at row %d, column %d" % (solution.word, play_direction, solution.row, solution.col))
    engine.runAndWait()
    print "Play %s %s at row %d, column %d" % (solution.word, play_direction, solution.row, solution.col)

if __name__ == "__main__":
    speak(solution)