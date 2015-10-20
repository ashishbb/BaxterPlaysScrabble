#!/usr/bin/env python

import sys
import rospy

from std_msgs.msg import String, Int32

def starter(num_blocks,configuration,command):
    pub1 = rospy.Publisher('num_blocks', Int32)
    pub2 = rospy.Publisher('configuration', String)
    pub3 = rospy.Publisher('command', String)
    rospy.init_node('starter')

    print "Number of Blocks: %s" %num_blocks
    print "configuration: %s" %configuration
    print "command: %s" %command
    pub1.publish(num_blocks)
    pub2.publish(configuration)
    pub3.publish(command)

if __name__ == "__main__":
    try:
        #get the number of blocks argument and verify it
        num_blocks = int(sys.argv[1])
        if num_blocks < 1:
            print "There is no way to manipulate %s blocks, assuming 1 block" %num_blocks
            num_blocks = 1
        #get the configuration argument and verify it
        configuration = (sys.argv[2])
        if (not (configuration == 'scattered' or configuration == \
            'stacked_ascending' or configuration == 'stacked_descending')):
            print "invalid configuration, assuming stacked_ascending"
            configuration = "stack_ascending"
        #get the command argument and verify it
        command = (sys.argv[3])
        if not (command == "scatter" or command == "stack_ascending" or command == "stack_descending" or command == "odd_even"):
            print "invalid command, assuming stack_descending"
            command = "stack_descending"
        starter(num_blocks,configuration,command)
    except rospy.ROSInterruptException:
        pass