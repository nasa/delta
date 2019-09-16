"""
Apply convert_input_image_folder.py using multiple PBS nodes to divide up the work.
"""
import os
import sys
import argparse
import traceback
import getpass
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from delta import pbs_functions #pylint: disable=C0413
# from delta.imagery import utilities #pylint: disable=C0413
import convert_input_image_folder #pylint: disable=C0413

#=========================================================================
# Parameters

# Constants used in this file

PBS_QUEUE   = 'normal'

GROUP_ID = 's2022'


PFE_NODES = ['san', 'ivy', 'has', 'bro']

#=========================================================================

# 'wes' = Westmere = 12 cores/24 processors, 48 GB mem, SBU 1.0, Launch from mfe1 only!
# 'san' = Sandy bridge = 16 cores,  32 GB mem, SBU 1.82
# 'ivy' = Ivy bridge   = 20 cores,  64 GB mem, SBU 2.52
# 'has' = Haswell      = 24 cores, 128 GB mem, SBU 3.34
# 'bro' = Broadwell    = 28 cores, 128 GB mem, SBU 4.04

def getParallelParams(nodeType):
    '''Return (numProcesses, tasksPerJob, maxHours) for running a certain task on a certain node type'''

    if nodeType == 'san': return ( 8, 200, 2)
    if nodeType == 'ivy': return (10, 300, 2)
    if nodeType == 'has': return (12, 400, 2)
    if nodeType == 'bro': return (14, 500, 2)
    if nodeType == 'wes': return ( 6, 200, 2)

    raise Exception('No params defined for node type ' + nodeType)

#=========================================================================

def getEmailAddress(user_name):
    '''Return the email address to use for a user'''

    if user_name == 'smcmich1':
        return 'scott.t.mcmichael@nasa.gov'
    raise Exception("Don't know email address for: " + user_name)


#---------------------------------------------------------------------

def write_input_files(file_list, files_per_job, output_prefix):
    """Create files containing lists of input file paths in random order where each is of length files_per_job.
       Returns the list of output files created."""

    num_files = len(file_list)
    indices   = np.arange(num_files)
    np.random.shuffle(indices)

    output_files = []
    input_iter = 0
    while input_iter < num_files:

        # Open the next output file
        current_file_count = 0
        file_path = output_prefix + str(len(output_files)) + '.txt'
        output_files.append(file_path)
        with open(file_path, 'w') as f:
            # Write files until we hit a limit
            while current_file_count < files_per_job:
                this_file = file_list[indices[input_iter]]
                f.write(this_file + '\n')

                current_file_count += 1
                input_iter += 1
                if input_iter >= num_files:
                    break

    return output_files


def submitBatchJobs(list_files, options, pass_along_args):
    '''Read all the batch jobs required for a run and distribute them across job submissions.
       Returns the common string in the job names.'''

    # Retrieve parallel processing parameters
    (numProcesses, tasksPerJob, maxHours) = getParallelParams(options.node_type)

    numBatches = len(list_files)
    print( ("Num batches: %d, tasks per job: %d" % (numBatches, tasksPerJob) ) )

    this_folder = os.path.dirname(os.path.realpath(__file__)) # won't change, unlike syspath
    scriptPath  = os.path.join(this_folder, 'convert_input_image_folder.py')

    # TODO: How to get this information?
    setup_commands = ['source /home1/smcmich1/software_build_dir/miniconda3/bin/activate',
                      'conda activate tf_112_cpu']

    index  = 0
    jobIDs = []
    for list_file in list_files:

        job_name = ('%s%05d' % ('DELTA_CI', index) )
        log_prefix = os.path.join(options.output_folder, job_name)

        # Specify the range of lines in the file we want this node to execute
        args = ('--input-file-list %s  --num-processes %d ' % \
                (list_file, numProcesses))
        args += ' '.join(pass_along_args)

        print('Submitting summary regen job: ' + scriptPath + ' ' + args)

        BATCH_PBS_QUEUE = 'normal'
        jobID = pbs_functions.submitJob(job_name, BATCH_PBS_QUEUE, maxHours,
                                        options.minutesInDevelQueue,
                                        GROUP_ID,
                                        options.node_type, 'python3',
                                        scriptPath + ' ' + args, log_prefix,
                                        priority=None,
                                        setup_commands=setup_commands)

        jobIDs.append(jobID)
        index += 1

    # Waiting on these jobs happens outside this function
    return jobIDs


def main(argsIn):

    try:
        usage = '''usage: convert_input_images_pbs.py <options> '''
        parser = argparse.ArgumentParser(usage=usage)

        parser.add_argument("--input-folder", dest="input_folder", default=None,
                            help="Path to the folder containing compressed images.")

        parser.add_argument("--output-folder", dest="output_folder", required=True,
                            help="Where to write the converted output images.")

        parser.add_argument("--input-file-list", dest="input_file_list", default=None,
                            help="Path to file listing all of the compressed image paths.")

        parser.add_argument("--image-type", dest="image_type", required=True,
                            help="Specify the input image type [worldview, landsat, tif, rgba].")

        parser.add_argument("--extension", dest="input_extension", default=None,
                            help="Manually specify the input extension instead of using the default.")

        parser.add_argument("--node-type",  dest="node_type", default='san',
                            help="Node type to use (wes[mfe], san, ivy, has, bro)")

        # Debug option
        parser.add_argument('--minutes-in-devel-queue', dest='minutesInDevelQueue', type=int,
                            default=0,
                            help="If positive, submit to the devel queue for this many minutes.")

        options, unknown = parser.parse_known_args(argsIn)

    except argparse.ArgumentError as msg:
        parser.error(msg)

#    if not utilities.checkIfToolExists('convert_input_image_folder.py'):
#        print("ERROR: Cannot run on PBS if the desired tool is not on $PATH")
#        return -1
    user_name = getpass.getuser()

    # Make sure our paths will work when called from PBS
    options.input_folder  = os.path.abspath(options.input_folder)
    options.output_folder = os.path.abspath(options.output_folder)

    # Get together all the CLI args that needs to be passed to each node
    pass_along_args = unknown
    pass_along_args += ['--input-folder', options.input_folder,
                        '--image-type', options.image_type,
                        '--output-folder', options.output_folder]

    input_file_list = convert_input_image_folder.get_input_files(options)
    print('Found ', len(input_file_list), ' input files to convert')
    if not input_file_list:
        return -1

    # Create input list files for each job
    if not os.path.exists(options.output_folder):
        os.mkdir(options.output_folder)
    list_file_prefix = os.path.join(options.output_folder, 'job_list_file-')
    os.system('rm ' + list_file_prefix + '*')
    files_per_job = getParallelParams(options.node_type)[1]
    job_inputs_file_list = write_input_files(input_file_list, files_per_job, list_file_prefix)
    print('Wrote ', len(job_inputs_file_list), ' input file lists')

    print("Disabling core dumps.") # these just take a lot of room
    os.system("ulimit -c 0")
    os.system("umask 022") # enforce files be readable by others

    try:

        # Call multi_process_command_runner.py through PBS for each chunk.
        jobIDs = submitBatchJobs(job_inputs_file_list, options, pass_along_args)

        # Wait for everything to finish.
        pbs_functions.waitForJobCompletion(jobIDs, user_name)

        resultText = 'All jobs are finished.'

    except Exception as e: #pylint: disable=W0703
        resultText = 'Caught exception: ' + str(e) + '\n' + traceback.format_exc()
    print('Result = ' + resultText)

    # Send a summary email.
    emailAddress = getEmailAddress(user_name)
    print("Sending email to: " + emailAddress)
    pbs_functions.send_email(emailAddress, 'PBS convert script finished!', resultText)

    print('Done with PBS batch convert script!')
    return 0


# Run main function if file used from shell
if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
