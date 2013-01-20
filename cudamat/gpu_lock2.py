#!/usr/bin/python

"""
A simple discretionary locking system for /dev/nvidia devices.

Iain Murray, November 2009, January 2010.

-- Additions -- Charlie Tang, Jan, 2011: 
added display of GPU usages

-- Charlie Tang, July, 2011:
improved statistics displaying
"""

import os
import os.path
from xml.dom import Node
from xml.dom.minidom import parseString
from subprocess import Popen, PIPE, STDOUT

_dev_prefix = '/dev/nvidia'

# Get ID's of NVIDIA boards. Should do this through a CUDA call, but this is
# a quick and dirty way that works for now:
def board_ids():
    """Returns integer board ids available on this machine."""
    #from glob import glob
    #board_devs = glob(_dev_prefix + '[0-9]*')
    #return range(len(board_devs))
    p = Popen(['/u/tang/bin/get_num_gpu_boards'], stdout=PIPE)    
    nBoards = int(p.stdout.read())
    return range(nBoards)

def _lock_file(id):
    """lock file from integer id"""
    # /tmp is cleared on reboot on many systems, but it doesn't have to be
    if os.path.exists('/dev/shm'):
        # /dev/shm on linux machines is a RAM disk, so is definitely cleared
        return '/dev/shm/gpu_lock_%d' % id
    else:
        return '/tmp/gpu_lock_%d' % id

def owner_of_lock(id):
    """Username that has locked the device id. (Empty string if no lock)."""
    import pwd
    try:
        statinfo = os.lstat(_lock_file(id))
        return pwd.getpwuid(statinfo.st_uid).pw_name
    except:
        return ""

def _obtain_lock(id):
    """Attempts to lock id, returning success as True/False."""
    try:
        # On POSIX systems symlink creation is atomic, so this should be a
        # robust locking operation:
        os.symlink('/dev/null', _lock_file(id))
        return True
    except:
        return False

def _launch_reaper(id, pid):
    """Start a process that will free a lock when process pid terminates"""
    from subprocess import Popen, PIPE
    me = __file__
    if me.endswith('.pyc'):
        me = me[:-1]
    myloc = os.path.dirname(me)
    if not myloc:
        myloc = os.getcwd()
    reaper_cmd = os.path.join(myloc, 'run_on_me_or_pid_quit')
    Popen([reaper_cmd, str(pid), me, '--free', str(id)],
        stdout=open('/dev/null', 'w'))

def obtain_lock_id(pid=None):
    """
    Finds a free id, locks it and returns integer id, or -1 if none free.

    A process is spawned that will free the lock automatically when the
    process pid (by default the current python process) terminates.
    """
    id = -1
    id = obtain_lock_id_to_hog()
    try:
        if id >= 0:
            if pid is None:
                pid = os.getpid()
            _launch_reaper(id, pid)
    except:
        free_lock(id)
        id = -1
    return id

def obtain_lock_id_to_hog():
    """
    Finds a free id, locks it and returns integer id, or -1 if none free.

    * Lock must be freed manually *
    """
    for id in board_ids():
        if _obtain_lock(id):
            return id
    return -1

def free_lock(id):
    """Attempts to free lock id, returning success as True/False."""
    try:
        filename = _lock_file(id)
        # On POSIX systems os.rename is an atomic operation, so this is the safe
        # way to delete a lock:
        os.rename(filename, filename + '.redundant')
        os.remove(filename + '.redundant')
        return True
    except:
        return False

def nvidia_gpu_stats():    
    p = Popen(['nvidia-smi', '-x', '-a'], stdout=PIPE)    
    output = p.stdout.read().lstrip()
    try:
        doc = parseString(output)
        gpucounter = 0        
        templist = []
        memlist = []
        uselist = []        
        fanlist = []
        doc2 = doc.getElementsByTagName("nvidia_smi_log")[0]
        gpulist = doc2.getElementsByTagName("gpu")
        for gpu in gpulist:        
            temp = gpu.getElementsByTagName('temperature')[0]            
            temp2 = temp.getElementsByTagName('gpu_temp')[0]
            templist.append(str(temp2.firstChild.toxml()))            
            mem = gpu.getElementsByTagName('memory_usage')[0]               
            memtot = mem.getElementsByTagName('total')[0]
            memused = mem.getElementsByTagName('used')[0]
            memfree = mem.getElementsByTagName('free')[0]            
            memtot_str = str(memtot.firstChild.toxml())
            memused_str = str(memused.firstChild.toxml())
            memfree_str = str(memfree.firstChild.toxml())
            memtot_float = float(memtot_str[:-3])            
            memused_float = float(memused_str[:-3])
            memfree_float = float(memfree_str[:-3])
            memlist.append('%03.f' % memused_float + '+%03.f' % memfree_float + '=%03.f' % memtot_float + 'Mb')
            use = gpu.getElementsByTagName('gpu_util')[0]        
            uselist.append(str(use.firstChild.toxml()))
            fan = gpu.getElementsByTagName('fan_speed')[0]
            fanlist.append(str(fan.firstChild.toxml()))
            gpucounter += 1
                    
        return [uselist, memlist, fanlist, templist]
    except:        
        return [ [-9999] * len(board_ids()) ] *4
       
         
# If run as a program:
if __name__ == "__main__":
    
    div = '  ' + "-" * 90    
    import sys
    me = sys.argv[0]
    # Report
    if '--id' in sys.argv:
        if len(sys.argv) > 2:
            try:
                pid = int(sys.argv[2])
                assert(os.path.exists('/proc/%d' % pid))
            except:
                print 'Usage: %s --id [pid_to_wait_on]' % me
                print 'The optional process id must exist if specified.'
                print 'Otherwise the id of the parent process is used.'
                sys.exit(1)
        else:
            pid = os.getppid()
        print obtain_lock_id(pid)
    elif '--ids' in sys.argv:
        try:
            id = int(sys.argv[2])            
        except:
            print 'Usage: %s --ids [specific gpu id]' % me
            sys.exit(1)       
        if _obtain_lock(id):
            print id
        else:
            print - 1
    elif '--id-to-hog' in sys.argv:
        print obtain_lock_id_to_hog()
    elif '--free' in sys.argv:
        try:
            id = int(sys.argv[2])
        except:
            print 'Usage: %s --free <id>' % me
            sys.exit(1)
        if free_lock(id):
            print "Lock freed"
        else:
            owner = owner_of_lock(id)
            if owner:
                print "Failed to free lock id=%d owned by %s" % (id, owner)        
            else:
                print "Failed to free lock, but it wasn't actually set?"
    elif '--noverbose' in sys.argv:
        stats = nvidia_gpu_stats()        
        print div
        print "%s board users:" % 'abc'
        print div       
        for id in board_ids():         
            print "      Board %d {Use:%s; Mem:%s; Temp:%s}: %s" % (id, stats[0][id], stats[1][id], stats[2][id], owner_of_lock(id))
        print div + '\n'
    else:
        stats = nvidia_gpu_stats()
        print div      
        print '  Usage instructions:\n'        
        print '  To obtain and lock an id: %s --id' % me
        print '  The lock is automatically freed when the parent terminates'
        print
        print "  To get an id that won't be freed: %s --id-to-hog <id>" % me
        print "  To get a specific id: %s --ids <id>" % me        
        print                                                   
        print "  You *must* manually free these ids: %s --free <id>\n" % me
        print '  More info: http://www.cs.toronto.edu/~murray/code/gpu_monitoring/'
        print '  Report any problems to: tang@cs.toronto.edu'    
        print '\n' + div
        print "  NVIDIA board users:"
        print div
        for id in board_ids():         
            print "  Board %d {Use:%s; Mem(used+free=total): %s; Fan:%s; Temp:%s}: %s" % (id, stats[0][id], stats[1][id], stats[2][id], stats[3][id], owner_of_lock(id))
        print div + '\n'


