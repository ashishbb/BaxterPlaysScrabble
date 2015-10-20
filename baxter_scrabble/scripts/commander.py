#!/usr/bin/env python

import sys
import rospy

from std_msgs.msg import String, Int32

def commander(command):
    pub3 = rospy.Publisher('command', String)
    rospy.init_node('commander')

    print "command: %s" %command
    pub3.publish(command)

if __name__ == "__main__":
    try:
        command = (sys.argv[1])
        if not (command == "scatter" or command == "stack_ascending" or command == "stack_descending" or command == "odd_even"):
            print "invalid command, assuming stack_descending"
            command = "stack_descending"
        commander(command)
    except rospy.ROSInterruptException:
        pass