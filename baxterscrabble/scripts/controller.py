#!/usr/bin/env python

import sys
import rospy
from aldenandthechipmunks_proj1.srv import *
from aldenandthechipmunks_proj1.msg import State
from std_msgs.msg import String, Int32

robot_interface = None
numberofblocks = 0
command = 'No Current Command'
config = None

current_command = 'No Current Command'
pub = None
numberofarms = 1

def callback1(data):
    print "blocks: %s" % ', '.join(map(str, data.blocksbottom))
    print "stacks: %s" % ', '.join(map(str, data.blocksstack))

def callback2(data):
    global command
    command = data.data
    print 'controller :' + command
    pick_routine()

def callback3(data):
    global numberofblocks
    numberofblocks = data.data
    print 'controller : %s' %numberofblocks

def callback4(data):
    global config
    config = data.data
    print 'controller :' + config

def controller():
    #initialize subscriptions
    rospy.Subscriber('state',State, callback1)
    rospy.Subscriber('command',String, callback2)
    rospy.Subscriber('num_blocks',Int32, callback3)
    rospy.Subscriber('configuration',String, callback4)

    #initialzie publication
    global pub
    pub = rospy.Publisher('configuration', String, queue_size=10)
    #initialize node
    rospy.init_node('controller')
    #set the robot interface service
    global robot_interface
    robot_interface = rospy.ServiceProxy('robot_interface', move_robot)
    rate = rospy.Rate(.2)
    while not rospy.is_shutdown():
        rospy.spin()

def pick_routine():
    if config == 'stacked_ascending':
        if command == 'stack_ascending':
            print "Stack is already ascending"
            global current_command
            current_command = 'stack_ascending'
            return
        elif command == 'stack_descending':
           print "Reversing stack"
           asc_reverse_stack(bool(1),numberofblocks)
           global current_command
           current_command = 'stack_ascending'
           return
        elif command == 'scatter':
            print "Scattering"
            asc_scatter(bool(1),numberofblocks)
            global current_command
            current_command = 'scatter'
            return
        elif command == 'odd_even':
            print "Performing odd_even"
            odd_even_asc(bool(1),numberofblocks)
            global current_command
            current_command = 'odd_even'
            return
    elif config == 'stacked_descending':
        if command == 'stack_ascending':
            print "Reversing stack"
            des_reverse_stack(bool(1),numberofblocks)
            global current_command
            current_command = 'stack_ascending'
            return
        elif command == 'stack_descending':
            print "Stack is already descending"
            global current_command
            current_command = 'stack_descending'
            return
        elif command == 'scatter':
            print "Scattering"
            des_scatter(bool(1),numberofblocks)
            global current_command
            current_command = 'scatter'
            return
        elif command == 'odd_even':
            print "Performing odd_even"
            odd_even_des(bool(1),numberofblocks)
            global current_command
            current_command = 'odd_even'
            return
    elif config == 'scattered':
        if command == 'stack_ascending':
            print "Building an ascending stack"
            descatter_ascending(bool(1),numberofblocks)
            global current_command
            current_command = 'stack_ascending'
            return
        elif command == 'stack_descending':
            print "Building a descending stack"
            descatter_descending(bool(1),numberofblocks)
            global current_command
            current_command = 'stack_descending'
            return
        elif command == 'scatter':
            print "Blocks are already scattered"
            global current_command
            current_command = 'scatter'
            return
        elif command == 'odd_even':
            print "Creating odd even towers"
            descatter_ascending(bool(1),numberofblocks)
            odd_even_asc(bool(1),numberofblocks)
            global current_command
            current_command = 'odd_even'
            return
    

def odd_even_des(is_left,num_blocks):
    for i in range(1,3):
        rospy.wait_for_service('robot_interface')
        robot_interface(is_left,3, i)
        robot_interface(is_left,2, i)
        rospy.wait_for_service('robot_interface')
        robot_interface(is_left,4, i)
        rospy.wait_for_service('robot_interface')
        robot_interface(is_left,1, i)
        if (numberofarms > 1):  
            rospy.wait_for_service('robot_interface')
            robot_interface(is_left,5, 0)           #ARM ARM ARM SWITCH
            is_left = not is_left 
    for i in range(3,num_blocks+1):
        rospy.wait_for_service('robot_interface')
        robot_interface(is_left,3, i)
        robot_interface(is_left,2, i)
        rospy.wait_for_service('robot_interface')
        robot_interface(is_left,4, i-2)
        rospy.wait_for_service('robot_interface')
        robot_interface(is_left,1, i)
        if (numberofarms > 1):      
            rospy.wait_for_service('robot_interface')
            robot_interface(is_left,5, 0)           #ARM ARM ARM SWITCH
            is_left = not is_left        #ARM ARM ARM SWITCH

def odd_even_asc(is_left,num_blocks):
    for i in range(0,2):
        rospy.wait_for_service('robot_interface')
        robot_interface(is_left,3, num_blocks-i)
        robot_interface(is_left,2, num_blocks-i)
        rospy.wait_for_service('robot_interface')
        robot_interface(is_left,4, num_blocks-i)
        rospy.wait_for_service('robot_interface')
        robot_interface(is_left,1, num_blocks-i)
        if (numberofarms > 1):  
            rospy.wait_for_service('robot_interface')
            robot_interface(is_left,5, 0)           #ARM ARM ARM SWITCH
            is_left = not is_left 
    for i in range(2,num_blocks+1):
        rospy.wait_for_service('robot_interface')
        robot_interface(is_left,3, num_blocks-i)
        robot_interface(is_left,2, num_blocks-i)
        rospy.wait_for_service('robot_interface')
        robot_interface(is_left,4, num_blocks-i+2)
        rospy.wait_for_service('robot_interface')
        robot_interface(is_left,1, num_blocks-i)
        if (numberofarms > 1):      
            rospy.wait_for_service('robot_interface')
            robot_interface(is_left,5, 0)           #ARM ARM ARM SWITCH
            is_left = not is_left        #ARM ARM ARM SWITCH




def asc_reverse_stack(is_left,num_blocks): #ONLY WORKS FOR ASCENDING, MUST FIX
    for i in range(0,num_blocks):
        rospy.wait_for_service('robot_interface')
        robot_interface(is_left,3, num_blocks-i)
        rospy.wait_for_service('robot_interface')
        robot_interface(is_left,2, num_blocks-i)
        rospy.wait_for_service('robot_interface')
        if i == 0:
            robot_interface(is_left,4, (num_blocks))
        else:
            robot_interface(is_left,4, (num_blocks-i+1))
        rospy.wait_for_service('robot_interface')
        robot_interface(is_left,1, num_blocks-i)
        if (numberofarms > 1):      
            rospy.wait_for_service('robot_interface')
            robot_interface(is_left,5, 0)           #ARM ARM ARM SWITCH
            is_left = not is_left        #ARM ARM ARM SWITCH
    global pub
    pub.publish('stacked_descending')

def des_reverse_stack(is_left,num_blocks): 
    for i in range(1,num_blocks+1):
        rospy.wait_for_service('robot_interface')
        robot_interface(is_left,3, i)
        rospy.wait_for_service('robot_interface')
        robot_interface(is_left,2, i)
        rospy.wait_for_service('robot_interface')
        if i == 1:
            robot_interface(is_left,4, i)
        else:
            robot_interface(is_left,4, i-1)
        rospy.wait_for_service('robot_interface')
        robot_interface(is_left,1, i)
        if (numberofarms > 1):    
            rospy.wait_for_service('robot_interface')
            robot_interface(is_left,5, 0)           #ARM ARM ARM SWITCH
            is_left = not is_left        #ARM ARM ARM SWITCH
    global pub
    pub.publish('stacked_ascending')

# def hanoi(num_blocks, helper, source, target):
#   helper = 1 #table 0
#   source = 2 #table 0 + .15x
#   target = 3 #table 0 + .3x
#   def hanoi(num_blocks, helper, source, target):


#     if n > 0:
#         # move tower of size n - 1 to helper:
#         hanoi((num_blocks - 1), source, target, helper)
#         # move disk from source peg to target peg
#         if source:
#             target.append(source.pop())

#         # move tower of size n-1 from helper to target
#         hanoi((num_blocks - 1), helper, source, target)
        
# source = [4,3,2,1]
# target = []
# helper = []
# hanoi(len(source),source,helper,target)

# print source, helper, target

def asc_scatter(is_left,num_blocks): #ONLY WORKS FOR ASCENDING, MUST FIX
    for i in range(0,num_blocks):  
        rospy.wait_for_service('robot_interface')
        robot_interface(is_left,3, num_blocks-i)   #move to
        rospy.wait_for_service('robot_interface')
        robot_interface(is_left,2, num_blocks-i)   #close
        rospy.wait_for_service('robot_interface')
        #moves over itself >> creates new stack
        robot_interface(is_left,4, num_blocks-i)   #move over
        rospy.wait_for_service('robot_interface')
        robot_interface(is_left,1, num_blocks-i)   #open
        if (numberofarms > 1):  
            rospy.wait_for_service('robot_interface')
            robot_interface(is_left,5, 0)           #ARM ARM ARM SWITCH
            is_left = not is_left        #ARM ARM ARM SWITCH
    global pub
    pub.publish('scattered')

def des_scatter(is_left,num_blocks):
    for i in range(1,num_blocks+1):
        rospy.wait_for_service('robot_interface')
        robot_interface(is_left,3, i)
        rospy.wait_for_service('robot_interface')
        robot_interface(is_left,2, i)
        rospy.wait_for_service('robot_interface')
        #moves over itself >> creates new stack 
        robot_interface(is_left,4, i)
        rospy.wait_for_service('robot_interface')
        robot_interface(is_left,1, i)
        if (numberofarms > 1):      
            rospy.wait_for_service('robot_interface')
            robot_interface(is_left,5, 0)           #ARM ARM ARM SWITCH
            is_left = not is_left        #ARM ARM ARM SWITCH
    global pub
    pub.publish('scattered')

def descatter_ascending(is_left,num_blocks):
    for i in range(1,num_blocks+1):
        rospy.wait_for_service('robot_interface')
        robot_interface(is_left,3, i)
        rospy.wait_for_service('robot_interface')
        robot_interface(is_left,2, i)
        rospy.wait_for_service('robot_interface')
        robot_interface(is_left,4, i-1)
        rospy.wait_for_service('robot_interface')
        robot_interface(is_left,1, i)
        if (numberofarms > 1):      
            rospy.wait_for_service('robot_interface')
            robot_interface(is_left,5, 0)           #ARM ARM ARM SWITCH
            is_left = not is_left
    global pub
    pub.publish('stacked_ascending')

def descatter_descending(is_left,num_blocks):
    for i in range(0,num_blocks):
        rospy.wait_for_service('robot_interface')
        robot_interface(is_left,3, (num_blocks-i))
        rospy.wait_for_service('robot_interface')
        robot_interface(is_left,2, (num_blocks-i))
        rospy.wait_for_service('robot_interface')
        robot_interface(is_left,4, (num_blocks-i+1)%(num_blocks+1))
        rospy.wait_for_service('robot_interface')
        robot_interface(is_left,1, (num_blocks-i))
        if (numberofarms > 1):      
            rospy.wait_for_service('robot_interface')
            robot_interface(is_left,5, 0)           #ARM ARM ARM SWITCH
            is_left = not is_left
    global pub
    pub.publish('stacked_descending')

if __name__ == "__main__":
    try:
        global numberofarms
        numberofarms = int(sys.argv[1])
        controller()
    except rospy.ROSInterruptException:
        pass