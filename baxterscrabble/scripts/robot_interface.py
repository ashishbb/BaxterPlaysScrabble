#!/usr/bin/env python

from aldenandthechipmunks_proj1.srv import move_robotResponse
from aldenandthechipmunks_proj1.srv import move_robot
from aldenandthechipmunks_proj1.srv import move_robotRequest
from aldenandthechipmunks_proj1.msg import State

import sys
import rospy
import baxter_interface

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import Header, String, Int32

from baxter_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)

rinumberofblocks = 0
riconfig = None

#starting position of the endpoint
startingpose = None
#initialize global arm and gripper
leftarm = None
rightarm = None
leftgripper = None
rightgripper = None
blocklength = .04445
#symbolic relationships of blocks
symbolic = bool(0)
blocksbot = []
blocksstk = []
#position of bottom of the block
blocksposx = []
blocksposy = []
blocksposz = []
#initialize held block
heldblock = -1
stacks = 1

#Handles calls on the robot_interface service
def handle_robot_interface(req):
    if req.target > rinumberofblocks:
        return bool(0)
    if not req.isleftarm:
        if req.action == 1:
            return open_gripper(rightgripper,req.target)
        elif req.action == 2:
            return close_gripper(rightgripper,req.target)  #hardcoded arm
        elif req.action == 3:
            return move_to_block(rightarm,req.target)
        elif req.action == 4:
            return move_over_block(rightarm,req.target)
        elif req.action == 5:
            if not symbolic:
                move_away(rightarm)
    if req.isleftarm:
        if req.action == 1:
            return open_gripper(leftgripper,req.target)
        elif req.action == 2:
            return close_gripper(leftgripper,req.target)  #hardcoded arm
        elif req.action == 3:
            return move_to_block(leftarm,req.target)
        elif req.action == 4:
            return move_over_block(leftarm,req.target)
        elif req.action == 5:
            if not symbolic:
                move_away(leftarm)
    return bool(0)

def open_gripper(gripper,target):
    if not symbolic:
        gripper.open()
        rospy.Rate(1).sleep()
    global heldblock
    heldblock = -1
    return bool(1)

def close_gripper(gripper,target):
    if not symbolic:
        gripper.close()
        rospy.Rate(1).sleep()
    global heldblock 
    heldblock = target
    return bool(1)

def move_over_block (limb,block_num):
    valid = bool(1)
    if not symbolic:
        #first bring the robot to a safe height above the stack
        rospy.Rate(5).sleep()
        current_pose = limb.endpoint_pose()
        current_point = current_pose['position']
        highest_z = 0.0
        if current_point.z > blocksposz[block_num]:
            highest_z = current_point.z + 2 * blocklength
        if current_point.z <= blocksposz[block_num]:
            highest_z = blocksposz[block_num] + 2 * blocklength
        move_to_point(limb,current_point.x,current_point.y,highest_z)
        #before moving down over a block, move to a position at a safe height above it
    print "held block: %s" %heldblock
    print "block_num : %s" %block_num
    if heldblock != block_num:
        valid = (on_top(block_num) or block_num == 0)
        if not symbolic:
            move_to_point(limb,blocksposx[block_num],blocksposy[block_num],highest_z)
            move_to_point(limb,blocksposx[block_num],blocksposy[block_num],(blocksposz[block_num]+blocklength))
        global blocksbot, blocksstk, blocksposx, blocksposy, blocksposz
        blocksbot[heldblock] = block_num
        blocksstk[heldblock] = blocksstk[block_num]
        if not symbolic:
            blocksposx[heldblock] = blocksposx[block_num]
            blocksposy[heldblock] = blocksposy[block_num]
            blocksposz[heldblock] = blocksposz[block_num]+blocklength
    #in the case where you command a block to move over itself, it will go to a new stack
    else:
        print "new stack"
        print stacks
        valid = on_top(block_num)
        if not symbolic:
            move_to_point(limb,blocksposx[0]+get_stack_x(stacks),blocksposy[0]+get_stack_y(stacks),highest_z)
            move_to_point(limb,blocksposx[0]+get_stack_x(stacks),blocksposy[0]+get_stack_y(stacks),blocksposz[0]+blocklength) #to accomodate hitting table
        global blocksbot, blocksstk, blocksposx, blocksposy, blocksposz, stacks
        blocksbot[heldblock] = 0
        blocksstk[heldblock] = (stacks + 1)%10
        if not symbolic:
            blocksposx[heldblock] = blocksposx[0]+get_stack_x(stacks)
            blocksposy[heldblock] = blocksposy[0]+get_stack_y(stacks)
            blocksposz[heldblock] = blocksposz[0]+blocklength #different from the move_to_point
        stacks = (stacks + 1)%10
    return valid

def get_stack_y(stack_num):
    if stack_num < 3:
        return 0
    if stack_num == 4 or stack_num == 6 or stack_num == 8:
        return 0.13
    if stack_num == 5 or stack_num == 3 or stack_num == 7:
        return (-0.13)
    if stack_num == 9 or stack_num == 10:
        return 0

def get_stack_x(stack_num):
    if stack_num == 1 or stack_num == 5 or stack_num == 6:
        return .13
    if stack_num == 2 or stack_num == 7 or stack_num == 8:
        return (-0.13)
    if stack_num > 2 and stack_num < 5:
        return 0
    if stack_num == 9:
        return .23
    if stack_num == 10:
        return -.23

def move_to_block (limb,block_num):
    if not symbolic:
        #first bring the robot to a safe height above the stack
        rospy.Rate(5).sleep()
        current_pose = limb.endpoint_pose()
        current_point = current_pose['position']
        highest_z = 0.0
        if current_point.z > blocksposz[block_num]:
            highest_z = current_point.z + 2 * blocklength
        if current_point.z <= blocksposz[block_num]:
            highest_z = blocksposz[block_num] + 2 * blocklength
        move_to_point(limb,current_point.x,current_point.y,highest_z)
        #before moving down over a block, move to a position at a safe height above it
        move_to_point(limb,blocksposx[block_num],blocksposy[block_num],highest_z)
        move_to_point(limb,blocksposx[block_num],blocksposy[block_num],blocksposz[block_num]) #was working with .05
    return on_top(block_num)

def on_top(block):
    for i in range(len(blocksbot)):
        if blocksbot[i] == block:
            return bool(0)
        else:
            return bool(1)

def move_away(limb):
    rospy.Rate(5).sleep()
    current_pose = limb.endpoint_pose()
    current_point = current_pose['position']
    point = startingpose['position']
    if limb == leftarm:
        move_to_point(limb,current_point.x,current_point.y,point.z  + 2 * blocklength)
        move_to_point(limb,blocksposx[0],blocksposy[0]+.45,point.z + 2 * blocklength)
        move_to_point(limb,blocksposx[0],blocksposy[0]+.45,point.z)
    if limb == rightarm:
        move_to_point(limb,current_point.x,current_point.y,point.z  + 2 * blocklength)
        move_to_point(limb,blocksposx[0],blocksposy[0]-.45,point.z + 2 * blocklength)
        move_to_point(limb,blocksposx[0],blocksposy[0]-.45,point.z)

def move_to_point(limb,xt,yt,zt):
    if limb == leftarm:
        iksvc = rospy.ServiceProxy("ExternalTools/left/PositionKinematicsNode/IKService", SolvePositionIK)
        ikreq = SolvePositionIKRequest()
        rospy.wait_for_service("ExternalTools/left/PositionKinematicsNode/IKService")
    if limb == rightarm:
        iksvc = rospy.ServiceProxy("ExternalTools/right/PositionKinematicsNode/IKService", SolvePositionIK)
        ikreq = SolvePositionIKRequest()
        rospy.wait_for_service("ExternalTools/right/PositionKinematicsNode/IKService")
    hdr = Header(stamp=rospy.Time.now(), frame_id='base')
    global startingpose
    newpose = PoseStamped(
            header=hdr,
            pose=Pose(
                position=Point(
                    x=1*xt, 
                    y=1*yt, 
                    z=1*zt
                ),
                orientation=startingpose['orientation'],
            ),
        )
    ikreq.pose_stamp.append(newpose)
    resp = iksvc(ikreq)
    limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
    limb.move_to_joint_positions(limb_joints)

def displace_by_vector(limb,xd,yd,zd):
    rospy.Rate(5).sleep()
    pose1 = limb.endpoint_pose()
    point = pose1['position']
    quaternion = pose1['orientation']
    newx = point.x + 1*xd
    newy = point.y + 1*yd
    newz = point.z + 1*zd
    move_to_point(limb,newx,newy,newz)

def init_block_positions(limb,num_blocks,configuration):
    if not symbolic:
        rospy.Rate(5).sleep()
        pose1 = limb.endpoint_pose()
        global startingpose
        startingpose = pose1
        point = pose1['position']
    global blocksbot, blocksstk, blocksposx, blocksposy, blocksposz
    blocksbot.append(-1)
    blocksstk.append(1)
    if not symbolic:
        blocksposx.append(point.x)
        blocksposy.append(point.y)
        blocksposz.append(point.z - num_blocks * blocklength)
    if configuration == 'stacked_ascending':
        for i in range(1,num_blocks+1):
            blocksbot.append(i-1)
            blocksstk.append(1)
            if not symbolic:
                blocksposx.append(point.x)
                blocksposy.append(point.y)
                blocksposz.append(point.z - (num_blocks-i) * blocklength)
    if configuration == 'stacked_descending':
        for i in range(1,num_blocks+1):
            blocksbot.append((i+1)%(num_blocks+1))
            blocksstk.append(1)
            if not symbolic:
                blocksposx.append(point.x)
                blocksposy.append(point.y)
                blocksposz.append(point.z - (i * blocklength))
    print "first blocks: %s" % ', '.join(map(str, blocksbot))
    print "first stacks: %s" % ', '.join(map(str, blocksstk))

def callback(data):
    global rinumberofblocks
    rinumberofblocks = data.data

def callback2(data):
    global riconfig
    riconfig = data.data


def robot_interface():
    #initialize this node as a publisher to state
    pub = rospy.Publisher('state', State, queue_size=10)
    rospy.Subscriber('configuration',String, callback2)
    rospy.Subscriber('num_blocks',Int32, callback)
    rospy.init_node('robot_interface')
    if not symbolic:
        #initializes limbs and grippers
        global leftarm, rightarm, leftgripper, rightgripper
        leftarm = baxter_interface.Limb('left')
        rightarm = baxter_interface.Limb('right')
        leftgripper = baxter_interface.Gripper('left')
        rightgripper = baxter_interface.Gripper('right')
    #initializes the state
    while riconfig == None or rinumberofblocks == 0:
        rospy.Rate(.5).sleep()
    init_block_positions(leftarm,rinumberofblocks,riconfig) #always start with this arm over block
    print "the ri config is: %s" %riconfig
    #handles the publishing of the state
    rate = rospy.Rate(1)
    s = rospy.Service('robot_interface', move_robot, handle_robot_interface)
    msg = State()
    msg.blocksbottom = blocksbot
    msg.blocksstack = blocksstk
    while not rospy.is_shutdown():
        pub.publish(msg)
        rate.sleep()

if __name__ == "__main__":
    try:
        global symbolic 
        symbolic = bool(int(sys.argv[1]))
        robot_interface()
    except rospy.ROSInterruptException:
        pass
