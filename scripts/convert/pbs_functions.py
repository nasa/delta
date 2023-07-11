# Copyright © 2020, United States Government, as represented by the
# Administrator of the National Aeronautics and Space Administration.
# All rights reserved.
#
# The DELTA (Deep Earth Learning, Tools, and Analysis) platform is
# licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Contains functions for working with the PBS job system on the Pleiades supercomputer'''

import os
import time
import subprocess
import re
import shlex

# Constants
MAX_PBS_NAME_LENGTH = 15

# Wait this many seconds between checking for job completion
SLEEP_TIME = 60

# The following functions are useful for going between string and list
#  representations of command line arguments
def isNotString(a):
    """Returns true if the object is not a string"""

    # Python 2/3 compatibilty
    try:
        basestring #pylint:disable=E0601
    except NameError:
        basestring = str

    return not isinstance(a, basestring)

def stringToArgList(string):
    """Converts a single argument string into a list of arguments"""
    return shlex.split(string)

def cleanJobID(jobID):
    '''Remove the part after the dot, when the input looks like 149691.pbspl233b.'''

    jobID = jobID.strip()
    m = re.match(r'^(.*?)\.', jobID)
    if m:
        return m.group(1)
    return jobID

def execute_command(cmd):
    """Simple replacement for the ASP command run function"""

    if not isNotString(cmd):
        cmd = stringToArgList(cmd)

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         universal_newlines=True)
    out, err = p.communicate()
    status = p.returncode
    return (out, err, status)

def send_email(address, subject, body):
    '''Send a simple email from the command line'''
    # Remove any quotes, as that confuses the command line.
    subject = subject.replace("\"", "")
    body    = body.replace("\"", "")
    try:
        cmd = 'mail -s "' + subject + '" ' + address + ' <<< "' + body + '"'
        #print(cmd) # too verbose to print
        os.system(cmd)
    except Exception: #pylint: disable=W0703
        print("Could not send mail.")

def getActiveJobs(user):
    '''Returns a list of the currently active jobs and their status'''

    # Run qstat command to get a list of all active jobs by this user
    cmd = ['qstat', '-u', user]

    (textOutput, _, status) = execute_command(cmd)

    lines = textOutput.split('\n')

    # Strip header lines
    NUM_HEADER_LINES = 3
    JOBID_INDEX      = 0
    NAME_INDEX       = 3
    STATUS_INDEX     = 7
    if len(lines) <= NUM_HEADER_LINES:
        return []
    lines = lines[NUM_HEADER_LINES:]

    # Pick out all the job names and put them in a list
    jobs = []
    for line in lines:
        parts = line.split()
        if len(parts) < STATUS_INDEX:
            continue
        jobID   = cleanJobID(parts[JOBID_INDEX])
        jobName = parts[NAME_INDEX]
        status  = parts[STATUS_INDEX]
        jobs.append((jobID, jobName, status))
    return jobs

def getNumCores(nodeType):
    '''Return the number of cores available for a given node type'''

    if nodeType == 'wes': # Merope
        return 12
    if nodeType == 'san':
        return 16
    if nodeType == 'ivy':
        return 20
    if nodeType == 'has':
        return 24
    if nodeType == 'bro':
        return 28
    raise Exception('Unrecognized node type: ' + nodeType)


# This is a less-capable version of the same function in ASP
def submitJob(jobName, queueName, maxHours, minutesInDevelQueue, #pylint: disable=R0913,R0914
              groupId, nodeType, commandPath, args, logPrefix, priority, setup_commands=None):
    '''Submits a job to the PBS system.'''
    setup_commands = setup_commands or []

    if len(queueName) > MAX_PBS_NAME_LENGTH:
        raise Exception('Job name "'+queueName+'" exceeds the maximum length of ' + str(MAX_PBS_NAME_LENGTH))

    numCpus = getNumCores(nodeType) # Cores or CPUs?

    hourString = '"'+str(maxHours)+':00:00"'

    # We must create and execute a shell script, to be able to explicitely
    # direct the output and error to files, without counting
    # on PBS to manage that, as that one overfloes.
    # All our outputs and errors will go to the "verbose" files,
    # while PBS will write its summaries, etc, to the other files.
    shellScriptPath   = logPrefix + '_script.sh'
    verboseOutputPath = logPrefix + '_verbose_output.log'
    verboseErrorsPath = logPrefix + '_verbose_errors.log'
    outputPath        = logPrefix + '_output.log'
    errorsPath        = logPrefix + '_errors.log'

    workDir = os.getcwd()

    # For debugging
    if minutesInDevelQueue > 0:
        queueName = 'devel'
        hourString = '00:' + str(minutesInDevelQueue).zfill(2) + ':00'

    # The "-m eb" option sends the user an email when the process begins and when it ends.
    # The -r n ensures the job does not restart if it runs out of memory.

    # Debug the environment
    #for v in os.environ.keys():
    #  logger.info("env is " + v + '=' + str(os.environ[v]))

    # We empty PYTHONSTARTUP and LD_LIBRARY_PATH so that python can function
    # properly on the nodes.
    priorityString = ''
    if priority:
        priorityString = ' -p ' + str(priority) + ' '

    # Generate the shell command
    shellCommand = ( "%s %s > %s 2> %s\n" % (commandPath, args,
                                             verboseOutputPath, verboseErrorsPath) )
    with open(shellScriptPath, 'w') as f:
        f.write("#!/bin/bash\n")
        for c in setup_commands:
            f.write(c + "\n")
        f.write(shellCommand)
    # Make it executable
    os.system("chmod a+rx " + shellScriptPath)

    # Run it
#    pbsCommand = ('qsub -r y -q %s -N %s %s -l walltime=%s -W group_list=%s -j oe -e %s -o %s -S /bin/bash -V -C %s -l select=1:ncpus=%d:model=%s  -- /usr/bin/env PYTHONPATH=%s PYTHONSTARTUP="" LD_LIBRARY_PATH="" %s' % #pylint: disable=C0301
#                  (queueName, jobName, priorityString, hourString, groupId,
#                   errorsPath, outputPath, workDir, numCpus, nodeType,
#                    pythonPath, shellScriptPath))
    pbsCommand = ('qsub -r y -q %s -N %s %s -l walltime=%s -W group_list=%s -j oe -e %s -o %s -S /bin/bash -V -C %s -l select=1:ncpus=%d:model=%s  -- /usr/bin/env  %s' % #pylint: disable=C0301
                  (queueName, jobName, priorityString, hourString, groupId,
                   errorsPath, outputPath, workDir, numCpus, nodeType, shellScriptPath))


    print(pbsCommand)
#    raise Exception('DEBUG')
    (out, err, status) = execute_command(pbsCommand)

    if status != 0:
        print(out)
        print(err)
        print("Status is: " + str(status))
        jobID = ''
    else:
        jobID = cleanJobID(out)

    print("Submitted job named " + str(jobName) + " with id " + str(jobID))

    return jobID


def waitForJobCompletion(jobIDs, user):
    '''Sleep until all of the submitted jobs containing the provided job prefix have completed'''

    print("Began waiting on " + str(len(jobIDs)) + " job(s)")

    jobsRunning = []
    stillWorking = True
    while stillWorking:

        time.sleep(SLEEP_TIME)
        stillWorking = False

        # Look through the list for jobs with the run's date in the name
        allJobs = getActiveJobs(user)

        numActiveJobs = 0
        for (jobID, jobName, status) in allJobs:
            if jobID in jobIDs:
                numActiveJobs += 1
                # Matching job found so we keep waiting
                stillWorking = True
                # Print a message if this is the first time we saw the job as running
                if (status == 'R') and (jobID not in jobsRunning):
                    jobsRunning.append(jobID)
                    print('Started running job named ' + str(jobName) + ' with id ' + str(jobID))

        print("Waiting on " + str(numActiveJobs) + " jobs")
